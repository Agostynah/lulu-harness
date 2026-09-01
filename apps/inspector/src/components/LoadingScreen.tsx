import { useEffect, useState } from "react";

// Parallax loading screen -- assets from Agos's own generation
// (public/loading/*.png): sky (clouds baked in -- the separate cloud
// layer she provided had magenta corruption baked into the cloud
// bodies, not just edge fringe, so it was dropped rather than papered
// over; see the conversation this was built in for the before/after),
// three grass depth layers scrolling at different speeds, and a
// 6-frame Lulu walk cycle. "LOADING" + the bar are NOT image assets --
// built in CSS so the fill percentage is real and themeable, matching
// what she asked for ("la barra se puede hacer programaticamente").
// Exported so App.tsx's unmount timer matches the CSS transition
// duration exactly (.loading-screen.exiting in styles.css) -- one
// number, not two that can drift apart.
export const LOADING_FADE_MS = 700;

// Every asset the scene draws, preloaded as a group before any of it is
// shown (see the loadedScene effect below). Without this, each
// background-image pops in independently as its own network/disk
// request finishes -- sky.jpg is the biggest so it visibly lagged
// behind the (smaller) cat sprite, arriving piecemeal instead of as one
// scene. Loading them explicitly via Image() and gating render on
// Promise.all means the whole scene fades in together, once.
const ASSET_URLS = [
  "/loading/sky.jpg",
  "/loading/grass-back.png",
  "/loading/grass-mid.png",
  "/loading/grass-front.png",
  "/loading/lulu-walk.png",
];

interface LoadingScreenProps {
  status: "connecting" | "error";
  onRetry?: () => void;
  fading?: boolean;
}

export default function LoadingScreen({ status, onRetry, fading = false }: LoadingScreenProps) {
  // Eases toward 90% while genuinely waiting, snaps to 100% only once
  // the caller reports done (status leaving "connecting" unmounts this
  // component from App.tsx, so 100% is never actually seen mid-flight --
  // this just keeps the fill from stalling if a connection takes a while).
  const [progress, setProgress] = useState(8);
  const [sceneLoaded, setSceneLoaded] = useState(false);

  useEffect(() => {
    if (status !== "connecting") return;
    const id = setInterval(() => {
      setProgress((p) => (p >= 90 ? 90 : p + (90 - p) * 0.08 + 0.4));
    }, 200);
    return () => clearInterval(id);
  }, [status]);

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      ASSET_URLS.map(
        (src) =>
          new Promise<void>((resolve) => {
            const img = new Image();
            img.onload = () => resolve();
            img.onerror = () => resolve(); // one broken asset shouldn't hang the whole scene
            img.src = src;
          })
      )
    ).then(() => {
      if (!cancelled) setSceneLoaded(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className={`loading-screen${fading ? " exiting" : ""}`}>
      <div className={`loading-scene${sceneLoaded ? " loaded" : ""}`}>
        <div className="parallax-sky" />
        <div className="parallax-layer parallax-grass-back" />
        <div className="parallax-layer parallax-grass-mid" />

        <div className="loading-hud">
          <span className="loading-title">LOADING</span>
          <div className="loading-bar-track">
            <div
              className="loading-bar-fill"
              style={{ width: status === "error" ? "100%" : `${progress}%` }}
            />
          </div>
          {status === "error" && (
            <div className="loading-error-box">
              <span>Can't reach lulu-server. Is it running? (`uv run lulu-server`)</span>
              {onRetry && (
                <button className="loading-retry" onClick={onRetry}>
                  Retry
                </button>
              )}
            </div>
          )}
        </div>

        <div className="parallax-cat" />
        <div className="parallax-layer parallax-grass-front" />
      </div>
    </div>
  );
}
