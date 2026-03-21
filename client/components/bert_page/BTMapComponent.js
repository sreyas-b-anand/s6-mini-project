"use client";
import { ReviewCard } from "@/components/Card";



export const BTMapComponent = ({ reviews }) => {
  if (!reviews || reviews.length === 0) {
    return (
      <p className="text-center text-muted">
        No reviews to display
      </p>
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