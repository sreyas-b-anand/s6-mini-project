"use client";
import { CloudAlert, CloudCheck } from "lucide-react";
import React from "react";

const MLResultComponent = ({ result }) => {
  if (!result) {
    return (
      <div className="rounded-md text-center max-w-xl mx-auto py-3 ">
        <p className="text-muted">Prediction will appear here.</p>
      </div>
    );
  }

  const isValid = result?.final_result === "Valid";

  return (
    <div className="rounded-md text-center max-w-xl mx-auto py-6 space-y-4">
      <p className="text-primary">The given review has been classified as :</p>
      <div
        className={`${
          isValid ? "text-green-600" : "text-red-600"
        } font-semibold text-lg flex items-center justify-center gap-3`}
      >
        {isValid ? <CloudCheck /> : <CloudAlert />}
        {isValid ? "Authentic Review" : "Fake Review"}
      </div>

      <p className="text-sm text-foreground/70">
        Confidence:{" "}
        <span className="font-semibold text-foreground">
          {(result?.final_confidence * 100).toFixed(2)}%
        </span>
      </p>
    </div>
  );
};

export default MLResultComponent;
