import { UI_TIERS, type UiTier } from "../types";

// Lets someone try a different tier's UI live without restarting the app
// or going back through OperatorSelect -- cycles basic -> advanced ->
// technomancer -> basic. Purely a display-complexity toggle (see
// types.ts's UiTier comment); doesn't touch the backend at all.
const TIER_LABELS: Record<UiTier, string> = {
  basic: "basic",
  advanced: "advanced",
  technomancer: "technomancer",
};

interface TierSwitcherProps {
  tier: UiTier;
  onChange: (tier: UiTier) => void;
}

export default function TierSwitcher({ tier, onChange }: TierSwitcherProps) {
  const cycle = () => {
    const i = UI_TIERS.indexOf(tier);
    onChange(UI_TIERS[(i + 1) % UI_TIERS.length]);
  };

  return (
    <button className="tier-switcher" title="Switch UI role (for testing)" onClick={cycle}>
      <RotateIcon />
      <span className="tier-switcher-label">role: {TIER_LABELS[tier]}</span>
    </button>
  );
}

function RotateIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
      <path
        d="M13.5 8A5.5 5.5 0 113 5.5"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
      <path d="M3 2.5V5.5H6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
