"use client";
import { ReviewCard } from "@/components/Card";
import { motion } from "framer-motion";

export const BTMapComponent = ({ reviews }) => {
  if (!reviews || !reviews.results || reviews.results.length === 0) {
    return (
      <div className="text-center py-6">
        <p className="text-muted text-sm">Predictions will appear here</p>
      </div>
    );
  }

  const fakePercentage = reviews.fake_percentage;

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-6"
    >
      <div className="surface p-5 space-y-3 text-center">
        <p className="text-sm text-muted">Extracted Reviews Analysis</p>

        {fakePercentage !== undefined && (
          <div className="space-y-2">
            <p className="text-foreground text-sm">Fake Review Ratio</p>

            <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${fakePercentage}%` }}
                transition={{ duration: 0.6 }}
                className="h-full bg-red-500"
              />
            </div>

            <p className="text-xs text-muted">
              <span className="font-semibold text-red-400">
                {fakePercentage}%
              </span>{" "}
              reviews predicted as fake
            </p>
          </div>
        )}
      </div>

      {/* Grid */}
      <div className="grid gap-4 md:grid-cols-2">
        {reviews.results.map((review, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <ReviewCard review={review} />
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};
