"use client";
import { CloudAlert, CloudCheck } from "lucide-react";
import React from "react";

const BTResultComponent = ({ result }) => {
  if (!result) {
    return (
      <div className="rounded-md text-center max-w-xl mx-auto py-0">
        <p className="text-muted">Prediction will appear here.</p>
      </div>
    );
  }


  const isValid = result.prediction === "OR"; 

  return (
    <div className="rounded-md text-center max-w-xl mx-auto py-0 space-y-4">

      <p className="text-primary">
        The review has been classified as.
      </p>

      <div
        className={`${
          isValid ? "text-green-600" : "text-red-600"
        } font-semibold text-lg flex items-center justify-center gap-3`}
      >
        {isValid ? <CloudCheck /> : <CloudAlert />}
        {isValid ? "Authentic Review" : "Fake Review"}
      </div>

      {result.confidence && (
        <p className="text-sm text-muted-foreground">
          Confidence:{" "}
          <span className="font-semibold text-foreground">
            {(result.confidence * 100).toFixed(2)}%
          </span>
        </p>
      )}

    </div>
  );
};

export default BTResultComponent;