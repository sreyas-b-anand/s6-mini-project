"use client";
import { ReviewCard } from "@/components/Card";



export const BTMapComponent = ({ reviews }) => {
  if (!reviews || reviews.length === 0) {
    return (
       <div className="w-full border rounded-md border-muted text-center max-w-xl mx-auto py-6">

      <p className="text-center text-muted">
        Predictions will appear here.
      </p>
       </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-2">
      {reviews.map((review, index) => (
        <ReviewCard key={index} review={review} />
      ))}
    </div>
  );
};