import { useEffect, useState } from "react";
import type { UiTier } from "../types";

// Shown once between the loading screen and the app itself (App.tsx
// skips straight past this on every later launch once a tier is picked
// -- see UI_TIER_STORAGE_KEY there). The parallax scene behind this
// screen reuses the exact same CSS classes LoadingScreen.tsx's assets
// use (.parallax-sky / .parallax-layer / .parallax-grass-*), just
// blurred -- so the loading screen doesn't visually "end" and restart,
// it stays in view, continuous, while this overlays on top of it.
const ASSET_URLS = [
  "/onboarding/title.webp",
  "/onboarding/basic_operator.webp",
  "/onboarding/system_technician.webp",
  "/onboarding/technomancer.webp",
];

interface OperatorSelectProps {
  onSelect: (tier: UiTier) => void;
}

export default function OperatorSelect({ onSelect }: OperatorSelectProps) {
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      ASSET_URLS.map(
        (src) =>
          new Promise<void>((resolve) => {
            const img = new Image();
            img.onload = () => resolve();
            img.onerror = () => resolve();
            img.src = src;
          })
      )
    ).then(() => {
      if (!cancelled) setLoaded(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="operator-select">
      <div className="operator-select-bg">
        <div className="parallax-sky" />
        <div className="parallax-layer parallax-grass-back" />
        <div className="parallax-layer parallax-grass-mid" />
        <div className="parallax-cat" />
        <div className="parallax-layer parallax-grass-front" />
      </div>
      <div className={`operator-select-content${loaded ? " loaded" : ""}`}>
        <img src="/onboarding/title.webp" alt="Welcome, authorized operator! Please select your system userrole:" className="operator-title" />
        <div className="operator-cards">
          <button className="operator-card" onClick={() => onSelect("basic")}>
            <img src="/onboarding/basic_operator.webp" alt="Basic Operator -- essential system functions, restricted permissions." />
          </button>
          <button className="operator-card" onClick={() => onSelect("advanced")}>
            <img src="/onboarding/system_technician.webp" alt="Systems Technician -- diagnostics and configuration, elevated permissions." />
          </button>
          <button className="operator-card" onClick={() => onSelect("technomancer")}>
            <img src="/onboarding/technomancer.webp" alt="Technomancer -- full system access and customization." />
          </button>
        </div>
      </div>
    </div>
  );
}
