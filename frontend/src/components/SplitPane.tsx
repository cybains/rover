import { ReactNode, useRef } from "react";

interface Props {
  left: ReactNode;
  right: ReactNode;
  atLive: boolean;
  onAtLiveChange: (at: boolean) => void;
}

export default function SplitPane({ left, right, atLive, onAtLiveChange }: Props) {
  const lRef = useRef<HTMLDivElement>(null);
  const rRef = useRef<HTMLDivElement>(null);

  const handle = (ref: React.RefObject<HTMLDivElement>) => {
    const el = ref.current!;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 5;
    onAtLiveChange(atBottom);
  };

  const jump = () => {
    if (lRef.current) lRef.current.scrollTop = lRef.current.scrollHeight;
    if (rRef.current) rRef.current.scrollTop = rRef.current.scrollHeight;
    onAtLiveChange(true);
  };

  return (
    <div style={{ display: "flex", height: "100%" }}>
      <div ref={lRef} onScroll={() => handle(lRef)} style={{ flex: 1, overflowY: "auto", padding: 8 }}>
        {left}
      </div>
      <div ref={rRef} onScroll={() => handle(rRef)} style={{ flex: 1, overflowY: "auto", padding: 8 }}>
        {right}
      </div>
      {!atLive && (
        <button style={{ position: "fixed", bottom: 20, right: 20 }} onClick={jump}>
          Jump to Live
        </button>
      )}
    </div>
  );
}
