"use client";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";

export default function QuizzesView() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      <div className="text-2xl font-semibold mb-4">Quizzes</div>
      <Card>
        <CardContent className="py-6 space-y-4">
          <div className="text-sm text-muted-foreground">Multiple-choice or open-ended. Difficulty selector.</div>
          <div className="flex items-center gap-2">
            <Select defaultValue="mcq">
              <SelectTrigger className="w-44"><SelectValue placeholder="Type" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="mcq">Multiple-choice</SelectItem>
                <SelectItem value="open">Open-ended</SelectItem>
              </SelectContent>
            </Select>
            <Select defaultValue="med">
              <SelectTrigger className="w-36"><SelectValue placeholder="Difficulty" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="easy">Easy</SelectItem>
                <SelectItem value="med">Medium</SelectItem>
                <SelectItem value="hard">Hard</SelectItem>
              </SelectContent>
            </Select>
            <Button>Generate</Button>
          </div>
          <div className="p-3 border rounded text-sm text-muted-foreground">(Quiz will appear here)</div>
        </CardContent>
      </Card>
    </div>
  );
}
