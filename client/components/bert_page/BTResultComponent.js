"use client";
import { CloudAlert, CloudCheck } from "lucide-react";
import React from "react";

const BTResultComponent = ({ result }) => {
  
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
          result.prediction === "OR" ? "text-green-600" : "text-red-600"
        } font-semibold text-lg flex items-center justify-center gap-3`}
      >
        {result.prediction === "OR" ? <CloudCheck /> : <CloudAlert />}
        {result.prediction === "OR" ? "Authentic Review" : "Fake Review"}
      </p>
    </div>
  );
};

export default BTResultComponent;
