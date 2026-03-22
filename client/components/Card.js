"use client";

import { Card, CardContent } from "@/components/ui/card";
import { CloudAlert, CloudCheck, Star } from "lucide-react";

export const ReviewCard = ({ review }) => {
  const isReal = review.prediction === "OR";

  return (
    <Card
      className={`w-full shadow-md border ${
        isReal ? "border-green-500" : "border-red-500"
      } hover:shadow-lg transition rounded-2xl ${
        isReal ? "bg-green-200/30" : "bg-red-200/30"
      }`}
    >
      <CardContent className="p-4 flex flex-col justify-between h-full">
        <div className="space-y-3">
          <p className="text-sm">{review.text}</p>
        </div>

        <div className="flex items-start gap-4 justify-between mt-4 flex-col">
          <div className="flex items-center gap-1 text-yellow-500">
            {[...Array(5)].map((_, i) => (
              <Star
                key={i}
                size={16}
                fill={i < review.rating ? "currentColor" : "none"}
              />
            ))}
            <span className="text-xs text-muted-foreground ml-2">
              {review.rating}/5
            </span>
          </div>

          <div className="w-full bg-muted h-2 rounded-full overflow-hidden">
            <div
              className={`h-2 ${isReal ? "bg-green-500" : "bg-red-500"}`}
              style={{ width: `${review.confidence * 100}%` }}
            />
          </div>
          <div
            className={`w-full flex items-center gap-2 font-semibold  justify-between ${
              isReal ? "text-green-600" : "text-red-600"
            }`}
          >
            <span className="flex items-center gap-2">
              {isReal ? <CloudCheck size={18} /> : <CloudAlert size={18} />}
              {isReal ? "Authentic" : "Fake"}
            </span>
            <span className="text-xs text-muted/80">
              {(review.confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
