"use client";
import { CloudAlert, CloudCheck } from "lucide-react";
import React from "react";
import { motion } from "framer-motion";

const MLResultComponent = ({ result }) => {
  if (!result) {
    return (
      <div className="text-center py-4">
        <p className="text-foreground text-sm">Prediction will appear here</p>
      </div>
    );
  }

  const isValid = result?.final_result === "Valid";
  const fakeProb = result?.final_confidence || 0;

  const displayConfidence = isValid
    ? ((1 - fakeProb) * 100).toFixed(2)
    : (fakeProb * 100).toFixed(2);

  const strength = result?.strength || "Uncertain";

  const getColorClasses = () => {
    switch (strength) {
      case "Strongly Valid":
        return {
          bg: "bg-green-500/10",
          text: "text-green-400",
          border: "border-green-500/20",
          bar: "bg-green-500",
        };
      case "Likely Valid":
        return {
          bg: "bg-green-400/10",
          text: "text-green-300",
          border: "border-green-400/20",
          bar: "bg-green-400",
        };
      case "Uncertain":
        return {
          bg: "bg-yellow-400/10",
          text: "text-yellow-500",
          border: "border-yellow-400/20",
          bar: "bg-yellow-400",
        };
      case "Likely Fake":
        return {
          bg: "bg-red-400/10",
          text: "text-red-400",
          border: "border-red-400/20",
          bar: "bg-red-400",
        };
      default: // Strongly Fake
        return {
          bg: "bg-red-500/10",
          text: "text-red-500",
          border: "border-red-500/20",
          bar: "bg-red-500",
        };
    }
  };

  const colors = getColorClasses();

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="surface p-6 space-y-5 text-center rounded-md "
    >
      <p className="text-md text-foreground">Prediction Result</p>

      <div
        className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium border ${colors.bg} ${colors.text} ${colors.border}`}
      >
        {isValid ? <CloudCheck size={18} /> : <CloudAlert size={18} />}
        {isValid ? "Authentic" : "Fake"} • {strength}
      </div>
      <p className="text-md text-foreground font-medium">
        {isValid
          ? "Low probability of being fake"
          : "High probability of being fake"}
      </p>
      <div className="space-y-2">
        <div className="flex  justify-center  gap-3 text-sm text-muted">
          <span>Probability : </span>
          <span className="text-foreground font-medium">
            {displayConfidence}%
          </span>
        </div>
      </div>
    </motion.div>
  );
};

export default MLResultComponent;
