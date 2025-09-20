"use client";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Sparkles } from "lucide-react";

export default function ExplainView() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      <div className="text-2xl font-semibold mb-4">Explain</div>
      <Card>
        <CardContent className="py-6 space-y-4">
          <div className="text-sm text-muted-foreground">Levels: Simple, Medium, Expert. Rewrites selected passage accordingly.</div>
          <div className="flex items-center gap-2">
            <Select defaultValue="simple">
              <SelectTrigger className="w-40"><SelectValue placeholder="Level" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="simple">Simple</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="expert">Expert</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="secondary"><Sparkles className="mr-2 h-4 w-4" />Rewrite</Button>
          </div>
          <div className="p-3 border rounded text-sm text-muted-foreground">(Explanation will appear here)</div>
        </CardContent>
      </Card>
    </div>
  );
}
