"use client";

import { Card, CardContent } from "@/components/ui/card";
import { CloudAlert, CloudCheck, Star } from "lucide-react";
import { motion } from "framer-motion";

export const ReviewCard = ({ review }) => {
  const isReal = review.prediction === "OR";
  const confidence = (review.confidence * 100).toFixed(1);

  return (
    <motion.div whileHover={{ y: -4 }} transition={{ duration: 0.2 }}>
      <Card className="surface card-hover h-full">
        <CardContent className="p-5 flex flex-col justify-between h-full space-y-4">
          {/* Review Text */}
          <p className="text-sm text-foreground leading-relaxed line-clamp-4">
            {review.text}
          </p>

          {/* Bottom Section */}
          <div className="space-y-4">
            {/* Rating */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1 text-yellow-400">
                {[...Array(5)].map((_, i) => (
                  <Star
                    key={i}
                    size={14}
                    fill={i < review.rating ? "currentColor" : "none"}
                  />
                ))}
              </div>

              <span className="text-xs text-muted">{review.rating}/5</span>
            </div>

            {/* Confidence Bar */}
            <div className="space-y-1">
              <div className="w-full h-1.5 bg-secondary rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${confidence}%` }}
                  transition={{ duration: 0.6 }}
                  className={`h-full ${isReal ? "bg-green-500" : "bg-red-500"}`}
                />
              </div>
            </div>

            {/* Result Row */}
            <div className="flex items-center justify-between text-xs">
              <div
                className={`flex items-center gap-1 font-medium ${
                  isReal ? "text-green-400" : "text-red-400"
                }`}
              >
                {isReal ? <CloudCheck size={16} /> : <CloudAlert size={16} />}
                {isReal ? "Authentic" : "Fake"}
              </div>

              <span className="text-muted">{confidence}%</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};
