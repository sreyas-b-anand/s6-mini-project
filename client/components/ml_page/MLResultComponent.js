"use client";
import { CloudAlert, CloudCheck } from "lucide-react";
import React from "react";

const MLResultComponent = ({ result }) => {
  if (!result) {
    return (
      <div className="border rounded-md border-muted text-center max-w-xl mx-auto py-6">
        <p className="text-muted">Prediction will appear here.</p>
      </div>
    );
  }

  return (
    <div className="border rounded-md border-muted text-center max-w-xl mx-auto py-6">
      <p
        className={`${
          result.prediction_ml === "Valid" ? "text-green-600" : "text-red-600"
        } font-semibold text-lg flex items-center justify-center gap-3`}
      >
        {result.prediction_ml === "Valid" ? <CloudCheck /> : <CloudAlert />}
        {result.prediction_ml === "Valid" ? "Authentic Review" : "Fake Review"}
      </p>
    </div>
  );
};

export default MLResultComponent;
