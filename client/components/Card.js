"use client";

import { Card, CardContent } from "@/components/ui/card";
import { CloudAlert, CloudCheck } from "lucide-react";


export const ReviewCard = ({ review }) => {
  const isReal = review.prediction === 1;

  return (
    <Card className={`w-full shadow-md ${isReal ? "border-green-600" : "border-red-600"} hover:shadow-lg transition rounded-2xl`}>
      <CardContent className="p-4 space-y-3">
        
        <p className="text-sm ">
          {review.text}
        </p>

        <div className="flex items-center justify-between">
          <div
            className={`flex items-center gap-2 font-semibold ${
              isReal ? "text-green-600" : "text-red-600"
            }`}
          >
            {isReal ? <CloudCheck size={18} /> : <CloudAlert size={18} />}
            {isReal ? "Authentic" : "Fake"}
          </div>

          <span className="text-xs text-muted/80">
            {(review.confidence * 100).toFixed(1)}%
          </span>
        </div>

        <div className="w-full bg-muted h-2 rounded-full overflow-hidden">
          <div
            className={`h-2 ${
              isReal ? "bg-green-500" : "bg-red-500"
            }`}
            style={{ width: `${review.confidence * 100}%` }}
          />
        </div>
      </CardContent>
    </Card>
  );
};