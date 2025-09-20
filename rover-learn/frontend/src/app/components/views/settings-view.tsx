"use client";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";

export default function SettingsView() {
  return (
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
  );
}
