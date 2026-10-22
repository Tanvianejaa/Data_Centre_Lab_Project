import React, { useState, useEffect, useLayoutEffect, useRef, useCallback } from 'react';
import confetti from 'canvas-confetti';
import { Volume2, VolumeX } from 'lucide-react';
import {
  playTick,
  playLock,
  playAiLock,
  playCorrect,
  playWrong,
  playWin
} from './utils/sound';

const DIFFICULTIES = [
  { id: 'easy', label: 'Easy', blurb: 'Half-blind. You should win this.' },
  { id: 'medium', label: 'Medium', blurb: 'A real contest.' },
  { id: 'hard', label: 'Hard', blurb: 'Sees what you see. Rarely misses.' }
];

const TARGETS = [3, 5, 10];
const TIME_LIMIT = 3.5;
const REVEAL_MS = 1500;
// Reading four labels costs a beat, and charging the player's clock for it
// handed the AI a head start it had not earned. The options are shown on their
// own first; the photo and the countdown only start once they are in place.
const READY_MS = 1800;
const GLIDE_MS = 620;

function Score({ who, value, tone }) {
  return (
    <div className="flex flex-col items-center leading-none">
      <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
        {who}
      </span>
      <span className={`tnum text-3xl font-black ${tone}`}>{value}</span>
    </div>
  );
}

export default function App() {
  const [screen, setScreen] = useState('lobby'); // lobby | playing | revealed | gameover
  const [difficulty, setDifficulty] = useState('medium');
  const [targetScore, setTargetScore] = useState(5);
  const [soundEnabled, setSoundEnabled] = useState(true);

  const [humanScore, setHumanScore] = useState(0);
  const [aiScore, setAiScore] = useState(0);
  const [roundNumber, setRoundNumber] = useState(0);
  const [humanStreak, setHumanStreak] = useState(0);
  const [aiStreak, setAiStreak] = useState(0);

  const [round, setRound] = useState(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [humanChoice, setHumanChoice] = useState(null);
  const [aiLocked, setAiLocked] = useState(false);
  const [reveal, setReveal] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [confirmQuit, setConfirmQuit] = useState(false);

  // --- refs -----------------------------------------------------------------
  // The countdown must not re-render the tree 60 times a second, and nothing
  // the loop reads may come from a closure that can go stale mid-round.
  const rafRef = useRef(null);
  const timeoutsRef = useRef([]);
  const startedAtRef = useRef(0);
  const lastTickRef = useRef(null);
  const clockRef = useRef(null); // DOM node for the numeric readout
  const barRef = useRef(null); // DOM node for the draining bar

  const roundRef = useRef(null);
  const humanRef = useRef(null);
  const aiRef = useRef(false);
  const soundRef = useRef(soundEnabled);
  const seqRef = useRef(0); // monotonic: anything resolving late checks this
  const screenRef = useRef('lobby');
  const optionsRef = useRef(null); // the answers, used to measure the travel
  const panelRef = useRef(null); // score + answers, what actually moves

  useEffect(() => {
    soundRef.current = soundEnabled;
  }, [soundEnabled]);

  useEffect(() => {
    screenRef.current = screen;
  }, [screen]);

  /**
   * Glide the answers in from the middle of the screen.
   *
   * They stay in their final slots the whole time and the whole group is moved
   * with one transform, so nothing re-lays-out and no card can cross another.
   * An earlier version laid them out horizontally and morphed each card's
   * position AND size into place; a row turning into a column makes the cards
   * pass through each other, and the shrinking boxes clipped the labels.
   */
  useLayoutEffect(() => {
    const panel = panelRef.current;
    const answers = optionsRef.current;
    if (!panel || !answers) return;
    if (screen === 'preview') {
      panel.style.transition = 'none';
      panel.style.transform = 'none';
      // Centre on the answers, not on the panel: the panel is a tall column
      // and centring its box would push the answers low. Moving the panel by
      // the answers' delta keeps the score directly above them.
      const r = answers.getBoundingClientRect();
      const dx = window.innerWidth / 2 - (r.left + r.width / 2);
      const dy = window.innerHeight / 2 - (r.top + r.height / 2);
      panel.style.zIndex = '20';
      panel.style.transform = `translate(${dx}px, ${dy}px) scale(1.06)`;
    } else if (screen === 'playing') {
      panel.style.transition = `transform ${GLIDE_MS}ms cubic-bezier(0.22, 0.75, 0.25, 1)`;
      panel.style.transform = 'none';
      const done = setTimeout(() => {
        if (panel) panel.style.zIndex = '';
      }, GLIDE_MS + 20);
      return () => clearTimeout(done);
    }
  }, [screen, round]);

  // Phase flags. Declared before the effects and handlers that close over them.
  // `reading` is the beat where the answers are centred and the photo is held
  // back, so the clock never charges you for reading the labels.
  const reading = screen === 'preview';
  const inRound = screen === 'preview' || screen === 'playing' || screen === 'revealed';

  const sound = (fn) => {
    if (soundRef.current) {
      try {
        fn();
      } catch {
        /* audio is non-essential */
      }
    }
  };

  const clearTimers = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    timeoutsRef.current.forEach(clearTimeout);
    timeoutsRef.current = [];
  }, []);

  const later = (fn, ms) => {
    timeoutsRef.current.push(setTimeout(fn, ms));
  };

  useEffect(() => clearTimers, [clearTimers]);

  // --- round lifecycle ------------------------------------------------------

  // Stop the draining bar where it stands, so an early lock visibly banks the
  // time you had left instead of the bar running on under the reveal.
  const freezeBar = () => {
    const bar = barRef.current;
    if (!bar) return;
    const scale = getComputedStyle(bar).transform;
    bar.style.transition = 'none';
    bar.style.transform = scale === 'none' ? 'scaleX(0)' : scale;
  };

  const submitRound = async (seq, choice, reactionTime, hScore, aScore, roundNum) => {
    clearTimers();
    freezeBar();
    const current = roundRef.current;
    if (seq !== seqRef.current || !current) return;

    try {
      const res = await fetch('/api/round/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          round_id: current.round_id,
          human_choice: choice,
          reaction_time: reactionTime
        })
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      if (seq !== seqRef.current) return;

      setReveal(data);
      setFrameIdx((current.frames?.length ?? 1) - 1);
      setScreen('revealed');
      sound(data.human.is_correct ? playCorrect : playWrong);

      setHumanStreak((n) => (data.human.is_correct ? n + 1 : 0));
      setAiStreak((n) => (data.ai.is_correct ? n + 1 : 0));

      const nextH = hScore + data.human.points_earned;
      const nextA = aScore + data.ai.points_earned;
      setHumanScore(nextH);
      setAiScore(nextA);

      if (nextH >= targetScore || nextA >= targetScore) {
        later(() => {
          setScreen('gameover');
          if (nextH >= targetScore) {
            sound(playWin);
            confetti({
              particleCount: 120,
              spread: 80,
              origin: { y: 0.6 },
              colors: ['#bf4d1f', '#15803d', '#1c1917', '#e0d8c9']
            });
          }
        }, REVEAL_MS);
      } else {
        later(() => startRound(roundNum + 1, nextH, nextA), REVEAL_MS);
      }
    } catch (err) {
      if (seq !== seqRef.current) return;
      console.error('Failed to submit round:', err);
      clearTimers();
      setError('Lost the round to a server error.');
    }
  };

  const runClock = (seq, aiDecisionTime, hScore, aScore, roundNum) => {
    startedAtRef.current = performance.now();
    lastTickRef.current = null;

    // The bar is a CSS transition, not a per-frame style write.
    if (barRef.current) {
      const bar = barRef.current;
      bar.style.transition = 'none';
      bar.style.transform = 'scaleX(1)';
      void bar.offsetWidth; // force the reset to land before the transition
      bar.style.transition = `transform ${TIME_LIMIT}s linear`;
      bar.style.transform = 'scaleX(0)';
    }

    // Discrete events are timeouts; only the text readout needs a frame loop.
    later(() => {
      if (seq !== seqRef.current || aiRef.current) return;
      aiRef.current = true;
      setAiLocked(true);
      sound(playAiLock);
    }, aiDecisionTime * 1000);

    later(() => {
      if (seq !== seqRef.current || humanRef.current) return;
      submitRound(seq, null, TIME_LIMIT, hScore, aScore, roundNum);
    }, TIME_LIMIT * 1000);

    const fractions = roundRef.current?.frame_fractions ?? [0, 0.3, 0.6, 0.9];
    fractions.forEach((f, i) => {
      if (i === 0) return;
      later(() => {
        if (seq === seqRef.current) setFrameIdx(i);
      }, f * TIME_LIMIT * 1000);
    });

    const tick = () => {
      if (seq !== seqRef.current) return;
      const elapsed = (performance.now() - startedAtRef.current) / 1000;
      const left = Math.max(0, TIME_LIMIT - elapsed);
      // Written straight to the DOM: no setState, so no re-render.
      if (clockRef.current) clockRef.current.textContent = left.toFixed(1);

      const whole = Math.ceil(left);
      if (whole !== lastTickRef.current && left > 0) {
        lastTickRef.current = whole;
        sound(playTick);
      }
      if (left > 0) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };

  const startRound = async (roundNum, hScore, aScore) => {
    clearTimers();
    const seq = ++seqRef.current;

    roundRef.current = null;
    humanRef.current = null;
    aiRef.current = false;

    setHumanChoice(null);
    setAiLocked(false);
    setReveal(null);
    setFrameIdx(0);
    setError(null);
    setRoundNumber(roundNum);
    setIsLoading(true);

    try {
      const res = await fetch('/api/round/next', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          category: 'all',
          difficulty,
          distortion_mode: 'auto',
          time_limit: TIME_LIMIT
        })
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      if (seq !== seqRef.current) return;

      roundRef.current = data;
      setRound(data);
      setIsLoading(false);
      setScreen('preview');
      later(() => {
        if (seq !== seqRef.current) return;
        setScreen('playing');
        runClock(seq, data.ai_decision_time, hScore, aScore, roundNum);
      }, READY_MS);
    } catch (err) {
      if (seq !== seqRef.current) return;
      console.error('Failed to load round:', err);
      setIsLoading(false);
      setError('Could not reach the game server.');
    }
  };

  const lockChoice = (choice) => {
    if (humanRef.current || screen !== 'playing') return;
    const reaction = Math.max(0.1, (performance.now() - startedAtRef.current) / 1000);
    humanRef.current = choice;
    setHumanChoice(choice);
    sound(playLock);
    submitRound(seqRef.current, choice, reaction, humanScore, aiScore, roundNumber);
  };

  const inPlayNow = () =>
    ['preview', 'playing', 'revealed'].includes(screenRef.current);

  const startMatch = () => {
    setHumanScore(0);
    setAiScore(0);
    setHumanStreak(0);
    setAiStreak(0);
    startRound(1, 0, 0);
  };

  const pauseForQuit = () => {
    if (!inPlayNow()) return;
    clearTimers();
    seqRef.current += 1; // pending timers and fetches now bail out
    setConfirmQuit(true);
  };

  // Resuming replays the round from scratch rather than handing back a clock
  // the player got to study the image during.
  const resume = () => {
    setConfirmQuit(false);
    startRound(roundNumber, humanScore, aiScore);
  };

  const toLobby = () => {
    clearTimers();
    seqRef.current += 1;
    roundRef.current = null;
    setConfirmQuit(false);
    setError(null);
    setIsLoading(false);
    setRound(null);
    setReveal(null);
    setScreen('lobby');
  };

  // --- keyboard -------------------------------------------------------------

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape' && inRound) {
        pauseForQuit();
        return;
      }
      if (screen !== 'playing' || !round || humanRef.current || confirmQuit) return;
      const i = parseInt(e.key, 10) - 1;
      if (i >= 0 && i < round.options.length) lockChoice(round.options[i]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [screen, round, confirmQuit, humanScore, aiScore, roundNumber]);

  // --- pieces ---------------------------------------------------------------

  const frameSrc = round?.frames?.[Math.min(frameIdx, (round.frames?.length ?? 1) - 1)];

  // Rendered above the photo on narrow screens and at the head of the answer
  // column on a laptop, so it is defined once and placed twice.
  const runOwner = humanStreak >= 2 ? 'you' : aiStreak >= 3 ? 'ai' : null;
  const runCount = runOwner === 'you' ? humanStreak : aiStreak;

  const scoreboard = (
    <div className="lg:pb-3 lg:border-b lg:border-rule">
      <div className="flex items-center justify-between px-1 lg:px-0">
        <Score who="You" value={humanScore} tone="text-clay" />
        <div className="text-center">
          <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
            Round <span className="tnum">{roundNumber}</span>
          </div>
          <div className="text-[11px] font-bold text-ink-faint tnum">to {targetScore}</div>
        </div>
        <Score who="AI" value={aiScore} tone="text-graphite" />
      </div>

      {/* Height is reserved so the board never shifts when a run starts. */}
      <div className="h-6 mt-1.5 flex items-center justify-center">
        {runOwner && (
          <span
            key={`${runOwner}-${runCount}`}
            className={`anim-pop px-2.5 py-1 rounded-full text-[10px] font-black uppercase
                        tracking-[0.12em] tnum ${
                          runOwner === 'you'
                            ? 'bg-clay-soft text-clay'
                            : 'bg-graphite-soft text-graphite'
                        }`}
          >
            {runOwner === 'you' ? `\u{1F525} ${runCount} in a row` : `AI ${runCount} in a row`}
          </span>
        )}
      </div>
    </div>
  );

  return (
    <div className="h-dvh flex flex-col overflow-hidden">
      <header className="shrink-0 border-b border-rule">
        <div className="max-w-xl mx-auto px-5 h-14 flex items-center justify-between">
          <button
            onClick={() => (inRound ? pauseForQuit() : toLobby())}
            className="text-[15px] font-black tracking-tight text-ink hover:text-clay transition-colors"
          >
            Vision Duel
          </button>

          <div className="flex items-center gap-2">
            {inRound && (
              <button
                onClick={pauseForQuit}
                className="px-3 py-1.5 rounded-lg text-[11px] font-bold uppercase tracking-wider
                           text-ink-soft border border-rule-strong hover:bg-graphite-soft transition-colors"
              >
                Stop
              </button>
            )}
            <button
              onClick={() => setSoundEnabled((s) => !s)}
              aria-label={soundEnabled ? 'Mute' : 'Unmute'}
              className="p-2 rounded-lg text-ink-soft hover:bg-graphite-soft transition-colors"
            >
              {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 min-h-0 flex flex-col items-center justify-center px-5 py-5 overflow-y-auto">
        {/* ---------------------------------------------------------- error */}
        {error && (
          <div className="anim-pop w-full max-w-sm bg-card border border-bad/30 rounded-2xl p-6 text-center space-y-4">
            <p className="font-bold text-bad">{error}</p>
            <div className="flex gap-2 justify-center">
              <button
                onClick={() => startRound(roundNumber, humanScore, aiScore)}
                className="px-4 py-2 rounded-xl bg-ink text-card font-bold text-sm hover:opacity-90"
              >
                Try again
              </button>
              <button
                onClick={toLobby}
                className="px-4 py-2 rounded-xl border border-rule-strong font-bold text-sm hover:bg-graphite-soft"
              >
                Quit
              </button>
            </div>
          </div>
        )}

        {/* -------------------------------------------------------- loading */}
        {isLoading && !error && (
          <div className="flex flex-col items-center gap-3 py-16">
            <div className="w-7 h-7 rounded-full border-2 border-rule-strong border-t-clay animate-spin" />
            <p className="text-xs font-semibold text-ink-faint uppercase tracking-wider">Next round</p>
          </div>
        )}

        {/* ---------------------------------------------------------- lobby */}
        {screen === 'lobby' && !error && !isLoading && (
          <div className="anim-rise w-full max-w-md space-y-9">
            <div className="text-center space-y-3">
              <h1 className="text-[44px] leading-[0.95] font-black tracking-tight">
                Name it
                <br />
                <span className="text-clay">before the machine.</span>
              </h1>
              <p className="text-sm text-ink-soft max-w-xs mx-auto leading-relaxed">
                A photo comes in scrambled and sharpens as the clock runs. Lock your answer
                before the AI locks its own.
              </p>
            </div>

            <div className="space-y-5">
              <fieldset>
                <legend className="text-[11px] font-bold uppercase tracking-[0.14em] text-ink-faint mb-2">
                  Playing to
                </legend>
                <div className="grid grid-cols-3 gap-2">
                  {TARGETS.map((pts) => (
                    <button
                      key={pts}
                      onClick={() => setTargetScore(pts)}
                      aria-label={`Play to ${pts} points`}
                      aria-pressed={targetScore === pts}
                      className={`py-3 rounded-xl border-2 font-bold transition-all ${
                        targetScore === pts
                          ? 'border-ink bg-ink text-card'
                          : 'border-rule-strong bg-card text-ink-soft hover:border-ink/40'
                      }`}
                    >
                      <span className="tnum text-lg">{pts}</span>
                      <span className="text-[11px] font-semibold"> pts</span>
                    </button>
                  ))}
                </div>
              </fieldset>

              <fieldset>
                <legend className="text-[11px] font-bold uppercase tracking-[0.14em] text-ink-faint mb-2">
                  Opponent
                </legend>
                <div className="grid grid-cols-3 gap-2">
                  {DIFFICULTIES.map((d) => (
                    <button
                      key={d.id}
                      onClick={() => setDifficulty(d.id)}
                      aria-pressed={difficulty === d.id}
                      className={`py-3 rounded-xl border-2 font-bold text-sm transition-all ${
                        difficulty === d.id
                          ? 'border-clay bg-clay text-card'
                          : 'border-rule-strong bg-card text-ink-soft hover:border-clay/40'
                      }`}
                    >
                      {d.label}
                    </button>
                  ))}
                </div>
                <p className="mt-2 text-xs text-ink-faint h-4">
                  {DIFFICULTIES.find((d) => d.id === difficulty)?.blurb}
                </p>
              </fieldset>
            </div>

            <button
              onClick={startMatch}
              className="w-full py-4 rounded-2xl bg-clay text-card font-black text-base tracking-wide
                         hover:brightness-110 active:scale-[0.99] transition-all shadow-sm"
            >
              Start
            </button>
          </div>
        )}

        {/* ----------------------------------------------------------- game */}
        {inRound && round && !error && !isLoading && (
          /* Portrait stack on narrow screens; on a laptop the photo takes the
             width it deserves and the answers sit in a column beside it. */
          <div
            className="w-full h-full max-w-md lg:max-w-[1000px] lg:max-h-[620px]
                       flex flex-col lg:flex-row gap-3 lg:gap-6"
          >
            <div className="shrink-0 lg:hidden">{scoreboard}</div>

            {/* Square photo. Width derives from the flex-resolved height, so the
                card is a true square on any screen; the lock strip sits outside
                it to keep that square exact. */}
            <div className="flex-1 min-h-0 min-w-0 flex flex-col items-center gap-2">
              <div
                className={`relative flex-1 min-h-0 aspect-square max-w-full rounded-2xl overflow-hidden
                            border shadow-sm transition-opacity duration-300 ${
                              reading
                                ? 'opacity-0 border-transparent bg-transparent'
                                : 'opacity-100 border-rule bg-graphite-soft'
                            }`}
              >
                {frameSrc && (
                  <img
                    src={frameSrc}
                    alt=""
                    className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${
                      reading ? 'opacity-0' : 'opacity-100'
                    }`}
                  />
                )}


                <div
                  className={`absolute top-0 inset-x-0 h-1.5 overflow-hidden transition-opacity ${
                    screen === 'preview' ? 'opacity-0' : 'bg-ink/15 opacity-100'
                  }`}
                >
                  <div
                    ref={barRef}
                    className={`h-full origin-left ${screen === 'revealed' ? 'bg-transparent' : 'bg-clay'}`}
                  />
                </div>

                {screen === 'playing' && (
                  <div
                    className="absolute top-4 right-4 px-3 py-1.5 rounded-xl bg-ink/75 backdrop-blur-sm
                               text-card text-base lg:text-lg font-black tnum"
                  >
                    <span ref={clockRef}>{TIME_LIMIT.toFixed(1)}</span>s
                  </div>
                )}

                {screen === 'revealed' && reveal && (
                  <div className="anim-pop absolute inset-x-0 bottom-0 bg-ink/85 backdrop-blur-sm px-4 py-3 text-center">
                    <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-card/60">
                      Answer
                    </div>
                    <div className="text-xl lg:text-2xl font-black text-card">
                      {reveal.correct_option}
                    </div>
                  </div>
                )}
              </div>

              {/* lock indicators */}
              <div className="shrink-0 w-full flex items-stretch rounded-xl overflow-hidden border border-rule bg-card text-[11px] lg:text-xs font-bold">
                <div
                  className={`flex-1 py-2 lg:py-2.5 text-center transition-colors ${
                    humanChoice ? 'bg-clay-soft text-clay' : 'text-ink-faint'
                  }`}
                >
                  {humanChoice ? 'You locked' : screen === 'preview' ? 'Read the options' : 'Pick one'}
                </div>
                <div className="w-px bg-rule" />
                <div
                  className={`flex-1 py-2 lg:py-2.5 text-center transition-colors ${
                    aiLocked ? 'bg-graphite-soft text-graphite' : 'text-ink-faint anim-think'
                  }`}
                >
                  {aiLocked ? 'AI locked' : screen === 'preview' ? 'AI waiting' : 'AI thinking'}
                </div>
              </div>
            </div>

            {/* answers */}
            <div
              ref={panelRef}
              className="shrink-0 lg:w-[340px] lg:shrink-0 flex flex-col gap-3 lg:gap-4
                         will-change-transform"
            >
              <div className="hidden lg:block shrink-0">{scoreboard}</div>

              <div
                ref={optionsRef}
                className="relative grid grid-cols-2 gap-2.5 lg:flex lg:flex-col lg:justify-center
                           lg:gap-3 lg:flex-1 lg:min-h-0"
              >
                {round.options.map((option, i) => {
                  const mine = humanChoice === option;
                  const theirs = reveal?.ai?.choice === option;
                  const right = reveal?.correct_option === option;

                  let style = 'bg-card border-rule-strong text-ink hover:border-ink/40 active:scale-[0.99]';
                  if (screen === 'revealed') {
                    if (right) style = 'bg-good-soft border-good text-good';
                    else if (mine) style = 'bg-bad-soft border-bad text-bad anim-shake';
                    else style = 'bg-card border-rule text-ink-faint';
                  } else if (mine) {
                    style = 'bg-clay-soft border-clay text-clay';
                  }

                  return (
                    <button
                      key={option}
                      disabled={!!humanChoice || screen !== 'playing'}
                      onClick={() => lockChoice(option)}
                      className={`relative min-h-[60px] lg:min-h-0 px-3 lg:px-4 py-3 lg:py-4 rounded-xl
                                  border-2 font-bold text-left transition-all disabled:cursor-default
                                  lg:flex lg:items-center lg:gap-3.5 ${style}`}
                    >
                      <span
                        className="block lg:shrink-0 lg:w-7 lg:h-7 lg:rounded-lg lg:border lg:border-current/25
                                   lg:grid lg:place-items-center text-[10px] lg:text-xs font-black
                                   text-ink-faint tnum mb-0.5 lg:mb-0"
                      >
                        {i + 1}
                      </span>
                      <span className="block text-sm lg:text-base leading-tight pr-10 lg:pr-0">
                        {option}
                      </span>

                      {screen === 'revealed' && (mine || theirs) && (
                        <span className="absolute top-2 right-2 lg:top-1/2 lg:-translate-y-1/2 flex gap-1">
                          {mine && (
                            <span className="px-1.5 py-0.5 rounded text-[9px] font-black uppercase bg-clay text-card">
                              You
                            </span>
                          )}
                          {theirs && (
                            <span className="px-1.5 py-0.5 rounded text-[9px] font-black uppercase bg-graphite text-card">
                              AI
                            </span>
                          )}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ------------------------------------------------------- game over */}
        {screen === 'gameover' && !error && (
          <div className="anim-rise w-full max-w-sm text-center space-y-7">
            <div className="space-y-2">
              <p className="text-[11px] font-bold uppercase tracking-[0.14em] text-ink-faint">
                {humanScore >= targetScore ? 'You win' : 'AI wins'}
              </p>
              <h2 className="text-[40px] leading-none font-black tracking-tight">
                {humanScore >= targetScore ? (
                  <span className="text-clay">Well seen.</span>
                ) : (
                  <span className="text-graphite">Out-looked.</span>
                )}
              </h2>
            </div>

            <div className="flex items-center justify-center gap-8 py-5 border-y border-rule">
              <Score who="You" value={humanScore} tone="text-clay" />
              <span className="text-ink-faint text-sm font-bold">—</span>
              <Score who="AI" value={aiScore} tone="text-graphite" />
            </div>

            <div className="flex gap-2.5">
              <button
                onClick={startMatch}
                className="flex-1 py-3.5 rounded-xl bg-clay text-card font-black hover:brightness-110 transition-all"
              >
                Rematch
              </button>
              <button
                onClick={toLobby}
                className="px-5 py-3.5 rounded-xl border-2 border-rule-strong font-bold text-ink-soft hover:bg-graphite-soft transition-colors"
              >
                Menu
              </button>
            </div>
          </div>
        )}
      </main>


      {/* ------------------------------------------------------ quit confirm */}
      {confirmQuit && (
        <div className="fixed inset-0 z-50 bg-ink/40 backdrop-blur-sm flex items-center justify-center p-5">
          <div className="anim-pop bg-card border border-rule-strong rounded-2xl p-6 w-full max-w-xs text-center space-y-5 shadow-xl">
            <div className="space-y-1">
              <p className="font-black text-lg">Stop the match?</p>
              <p className="text-xs text-ink-soft tnum">
                You {humanScore} — {aiScore} AI
              </p>
              <p className="text-[11px] text-ink-faint pt-1">
                Keeping going replays this round.
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={toLobby}
                className="flex-1 py-2.5 rounded-xl bg-ink text-card font-bold text-sm hover:opacity-90"
              >
                Stop
              </button>
              <button
                onClick={resume}
                className="flex-1 py-2.5 rounded-xl border-2 border-rule-strong font-bold text-sm hover:bg-graphite-soft"
              >
                Keep playing
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
