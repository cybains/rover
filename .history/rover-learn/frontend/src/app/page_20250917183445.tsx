import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Mic, Radio, Play, Square, Bookmark, Settings, Download, Cpu, Globe, PanelRightOpen, PanelRight, Timer, Clock, BookOpen, FolderDown, Highlighter, ListChecks } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";

// --- Mock Data
const sampleTranscript = [
  { id: 1, time: "00:00:02", text: "Guten Morgen, alle zusammen. Heute sprechen wir über lineare Regression.", speaker: "Dr. Müller", q: false },
  { id: 2, time: "00:00:08", text: "Was sind die Grundannahmen dieses Modells?", speaker: "Student", q: true },
  { id: 3, time: "00:00:12", text: "Die Annahmen umfassen Linearität, Unabhängigkeit, Homoskedastizität und Normalverteilung der Fehler.", speaker: "Dr. Müller", q: false },
];

const sampleTranslation = [
  { id: 1, time: "00:00:02", text: "Good morning, everyone. Today we will talk about linear regression.", speaker: "Dr. Müller", q: false },
  { id: 2, time: "00:00:08", text: "What are the basic assumptions of this model?", speaker: "Student", q: true },
  { id: 3, time: "00:00:12", text: "The assumptions include linearity, independence, homoskedasticity, and normal distribution of errors.", speaker: "Dr. Müller", q: false },
];

function LatencyPill({ ms }: { ms: number }) {
  const color = ms < 400 ? "bg-emerald-500" : ms < 800 ? "bg-amber-500" : "bg-rose-500";
  return (
    <div className={`flex items-center gap-2 rounded-full px-3 py-1 text-white ${color}`}>
      <Timer className="h-4 w-4" />
      <span className="text-sm">{ms} ms</span>
    </div>
  );
}

function StatusPill({ live }: { live: boolean }) {
  return (
    <Badge variant={live ? "default" : "secondary"} className="rounded-full px-3 py-1">
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${live ? "bg-emerald-500" : "bg-gray-400"}`} />
        {live ? "Live" : "Stopped"}
      </div>
    </Badge>
  );
}

function Pane({ title, items }: { title: string; items: typeof sampleTranscript }) {
  const listRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [items]);
  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-lg font-semibold">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div ref={listRef} className="h-[58vh] overflow-y-auto pr-2 space-y-3">
          {items.map((seg) => (
            <motion.div key={seg.id} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} className="group">
              <div className="flex items-start gap-3">
                <Badge variant="secondary" className="shrink-0">{seg.time}</Badge>
                <div className="flex-1">
                  <div className="text-sm text-muted-foreground">{seg.speaker}</div>
                  <p className={`leading-relaxed ${seg.q ? "border-l-2 pl-3 border-amber-400" : ""}`}>{seg.text}</p>
                </div>
                <div className="opacity-0 group-hover:opacity-100 transition">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button variant="ghost" size="icon"><Bookmark className="h-4 w-4" /></Button>
                      </TooltipTrigger>
                      <TooltipContent>Bookmark</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default function LearningAppUI() {
  const [live, setLive] = useState(false);
  const [latency, setLatency] = useState(320);
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeView, setActiveView] = useState<'session' | 'bookmarks' | 'settings' | 'exports'>("session");

  useEffect(() => {
    const id = setInterval(() => setLatency((l) => Math.max(120, Math.min(1200, Math.round(l + (Math.random() * 80 - 40))))), 1200);
    return () => clearInterval(id);
  }, []);

  // --- Left menu items
  const MenuButton = ({ label, icon: Icon, value }: { label: string; icon: any; value: 'session'|'bookmarks'|'settings'|'exports' }) => (
    <Button
      variant={value === 'session' ? 'secondary' : 'ghost'}
      className="w-full justify-start"
      onClick={() => { setActiveView(value); setMenuOpen(false); }}
    >
      <Icon className="mr-2 h-4 w-4" /> {label}
    </Button>
  );

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <div className="sticky top-0 z-20 backdrop-blur supports-[backdrop-filter]:bg-background/70 border-b">
        <div className="max-w-7xl mx-auto flex items-center justify-between px-4 py-3">
          <button className="flex items-center gap-3 group cursor-pointer" onClick={() => setMenuOpen(true)}>
            <div className="h-8 w-8 rounded-xl bg-foreground/90 text-background grid place-items-center font-bold group-hover:opacity-90">Λ</div>
            <div>
              <div className="font-semibold leading-tight group-hover:underline">Learning Lab</div>
              <div className="text-xs text-muted-foreground">Personal build · Dev mode</div>
            </div>
            <Badge variant="outline" className="ml-2">v0.1 UI</Badge>
          </button>

          <div className="flex items-center gap-3">
            <LatencyPill ms={latency} />
            <StatusPill live={live} />
          </div>
        </div>
      </div>

      {/* Left Menu */}
      <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
        <SheetContent side="left" className="w-[260px]">
          <SheetHeader>
            <SheetTitle>Main Menu</SheetTitle>
          </SheetHeader>
          <div className="mt-4 space-y-2">
            <MenuButton label="Start a Session" icon={Play} value="session" />
            <MenuButton label="Bookmarks" icon={Bookmark} value="bookmarks" />
            <MenuButton label="Settings" icon={Settings} value="settings" />
            <MenuButton label="Exports" icon={Download} value="exports" />
          </div>
        </SheetContent>
      </Sheet>

      {/* Views */}
      {activeView === 'session' && (
        <>
          {/* Controls (minimal placeholder to keep focus on panes) */}
          <div className="border-b">
            <div className="max-w-7xl mx-auto grid grid-cols-12 items-center gap-3 px-4 py-3">
              <div className="col-span-4 flex items-center gap-2">
                <Badge variant="secondary" className="gap-2"><Cpu className="h-3 w-3" /> Local</Badge>
                <Badge variant="secondary" className="gap-2"><Globe className="h-3 w-3" /> Offline-first</Badge>
              </div>
              <div className="col-span-4 flex items-center gap-2">
                <Select defaultValue="mic">
                  <SelectTrigger className="w-40"><SelectValue placeholder="Source" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mic"><div className="flex items-center gap-2"><Mic className="h-4 w-4" /> Microphone</div></SelectItem>
                    <SelectItem value="loopback"><div className="flex items-center gap-2"><Radio className="h-4 w-4" /> System Audio</div></SelectItem>
                  </SelectContent>
                </Select>
                <div className="flex items-center gap-2">
                  <Select defaultValue="de">
                    <SelectTrigger className="w-28"><SelectValue placeholder="Input" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="de">German</SelectItem>
                      <SelectItem value="en">English</SelectItem>
                      <SelectItem value="pt">Portuguese</SelectItem>
                      <SelectItem value="es">Spanish</SelectItem>
                    </SelectContent>
                  </Select>
                  <span className="text-muted-foreground">→</span>
                  <Select defaultValue="en">
                    <SelectTrigger className="w-28"><SelectValue placeholder="Output" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="en">English</SelectItem>
                      <SelectItem value="de">German</SelectItem>
                      <SelectItem value="pt">Portuguese</SelectItem>
                      <SelectItem value="es">Spanish</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="col-span-4 flex items-center justify-end gap-2">
                <Button onClick={() => setLive((v) => !v)}>
                  {live ? <><Square className="mr-2 h-4 w-4" /> Stop</> : <><Play className="mr-2 h-4 w-4" /> Start</>}
                </Button>
                <Button variant="outline"><Bookmark className="mr-2 h-4 w-4" /> Bookmark</Button>
              </div>
            </div>
          </div>

          {/* Main Content: Transcript + Translation side-by-side */}
          <div className="max-w-7xl mx-auto grid grid-cols-12 gap-4 px-4 py-4">
            <div className="col-span-12 lg:col-span-6">
              <Pane title="Transcript (DE)" items={sampleTranscript} />
            </div>
            <div className="col-span-12 lg:col-span-6">
              <Pane title="Translation (EN)" items={sampleTranslation} />
            </div>
          </div>

          {/* Timeline */}
          <div className="border-t">
            <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-3">
              <Clock className="h-4 w-4" />
              <div className="w-full">
                <Progress value={live ? Math.min(100, latency / 12) : 0} />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>00:00</span>
                  <span>00:45</span>
                </div>
              </div>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon"><Highlighter className="h-4 w-4" /></Button>
                  </TooltipTrigger>
                  <TooltipContent>Auto-highlight questions</TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon"><ListChecks className="h-4 w-4" /></Button>
                  </TooltipTrigger>
                  <TooltipContent>Enforce glossary</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
        </>
      )}

      {activeView === 'bookmarks' && (
        <div className="max-w-5xl mx-auto px-4 py-16">
          <div className="text-2xl font-semibold mb-4">Bookmarks</div>
          <Card><CardContent className="py-6 text-sm text-muted-foreground">Your saved moments will appear here.</CardContent></Card>
        </div>
      )}

      {activeView === 'settings' && (
        <div className="max-w-5xl mx-auto px-4 py-16">
          <div className="text-2xl font-semibold mb-4">Settings</div>
          <Card>
            <CardContent className="grid grid-cols-2 gap-4 py-6">
              <div>
                <div className="text-sm font-medium mb-1">Chunk Size (ms)</div>
                <Input placeholder="500" defaultValue={500} />
              </div>
              <div>
                <div className="text-sm font-medium mb-1">Silence Gate (ms)</div>
                <Input placeholder="120" defaultValue={120} />
              </div>
              <div className="col-span-2 flex items-center justify-between border rounded-lg p-3">
                <div>
                  <div className="text-sm font-medium">Force Input Language</div>
                  <div className="text-xs text-muted-foreground">Avoid auto-detect drift</div>
                </div>
                <Switch />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {activeView === 'exports' && (
        <div className="max-w-5xl mx-auto px-4 py-16">
          <div className="text-2xl font-semibold mb-4">Exports</div>
          <div className="grid grid-cols-2 gap-3">
            <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> JSONL</Button>
            <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> VTT / SRT</Button>
            <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> Transcripts</Button>
            <Button variant="secondary"><FolderDown className="mr-2 h-4 w-4" /> Study Pack</Button>
          </div>
          <div className="text-xs text-muted-foreground mt-3">Exports are local; nothing leaves your machine.</div>
        </div>
      )}

      {/* Footer */}
      <div className="border-t bg-muted/30">
        <div className="max-w-7xl mx-auto px-4 py-3 text-xs text-muted-foreground flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="font-medium">Hotkeys:</span>
            <span>Start/Stop ⌘/Ctrl+K</span>
            <span>Bookmark B</span>
            <span>Search ⌘/Ctrl+F</span>
          </div>
          <div>Built for: You · 100% local · Vienna</div>
        </div>
      </div>
    </div>
  );
}
