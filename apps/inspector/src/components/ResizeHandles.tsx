// decorations:false (tauri.conf.json) means the OS gives this window NO
// resize affordance at all -- move works via data-tauri-drag-region
// (TitleBar.tsx), but resize needs these explicit invisible edge/corner
// strips calling startResizeDragging(). Only rendered in the Tauri
// shell; the plain-browser dev flow already resizes via the OS/browser
// chrome and never mounts this.
const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

type Direction = "North" | "South" | "East" | "West" | "NorthEast" | "NorthWest" | "SouthEast" | "SouthWest";

const EDGES: { className: string; direction: Direction }[] = [
  { className: "resize-handle-n", direction: "North" },
  { className: "resize-handle-s", direction: "South" },
  { className: "resize-handle-e", direction: "East" },
  { className: "resize-handle-w", direction: "West" },
  { className: "resize-handle-ne", direction: "NorthEast" },
  { className: "resize-handle-nw", direction: "NorthWest" },
  { className: "resize-handle-se", direction: "SouthEast" },
  { className: "resize-handle-sw", direction: "SouthWest" },
];

export default function ResizeHandles() {
  if (!isTauri) return null;

  const startResize = (direction: Direction) => {
    import("@tauri-apps/api/window").then(({ getCurrentWindow }) => {
      getCurrentWindow().startResizeDragging(direction);
    });
  };

  return (
    <>
      {EDGES.map(({ className, direction }) => (
        <div
          key={className}
          className={`resize-handle ${className}`}
          onMouseDown={() => startResize(direction)}
        />
      ))}
    </>
  );
}
