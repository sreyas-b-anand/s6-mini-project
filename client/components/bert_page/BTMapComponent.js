"use client";
import { ReviewCard } from "@/components/Card";

export const BTMapComponent = ({ reviews }) => {
  if (!reviews || reviews.length === 0) {
    return (
      <div className="w-full  rounded-md  text-center max-w-xl mx-auto py-6">
        <p className="text-center text-muted">Predictions will appear here.</p>
      </div>
    );
  }

  return (
    <>
      <div>
        <p className="text-center text-primary mb-4">
          {reviews &&
            reviews.length > 0 &&
            "Below are the reviews extracted from the provided product link, along with their predicted authenticity."}
        </p>
        <div className="text-foreground/70 text-center">
          {reviews?.fake_percentage && (
            <p>
              Approximately{" "}
              <span className="font-bold text-primary">
                {reviews?.fake_percentage}%
              </span>{" "}
              of the reviews are predicted to be fake.
            </p>
          )}
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-2">
        {reviews &&
          reviews?.results?.map((review, index) => (
            <ReviewCard key={index} review={review} />
          ))}
      </div>
    </>
  );
};
