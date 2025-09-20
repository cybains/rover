"use client";

export default function FooterHotkeys() {
  return (
    <div className="border-t bg-muted/30">
      <div className="max-w-7xl mx-auto px-4 py-3 text-xs text-muted-foreground flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-medium">Hotkeys:</span>
          <span><kbd>⌘/Ctrl</kbd>+<kbd>K</kbd> Start/Resume</span>
          <span><kbd>B</kbd> Bookmark</span>
          <span><kbd>⌘/Ctrl</kbd>+<kbd>M</kbd> Open Menu</span>
        </div>
        <div>Built for: You · 100% local · Vienna</div>
      </div>
    </div>
  );
}
