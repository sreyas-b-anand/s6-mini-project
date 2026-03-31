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

  const fakeProb = result?.final_confidence || 0;
  const strength = result?.strength || "Strongly Fake";

  const strengthConfig = {
    "Strongly Valid": {
      label: "Valid",
      message: "This content is highly reliable and strongly authentic.",
      bg: "bg-green-500/10",
      text: "text-green-500",
      border: "border-green-500/20",
    },
    "Likely Valid": {
      label: "Valid",
      message: "This content appears to be reliable and likely authentic.",
      bg: "bg-green-400/10",
      text: "text-green-400",
      border: "border-green-400/20",
    },
    "Likely Fake": {
      label: "Fake",
      message: "This content is suspicious and likely to be fake.",
      bg: "bg-red-400/10",
      text: "text-red-400",
      border: "border-red-400/20",
    },
    "Strongly Fake": {
      label: "Fake",
      message: "This content is highly unreliable and strongly fake.",
      bg: "bg-red-500/10",
      text: "text-red-500",
      border: "border-red-500/20",
    },
  };

  const config = strengthConfig[strength] || strengthConfig["Strongly Fake"];

  const isValid = config.label === "Valid";

  const displayConfidence = isValid
    ? ((1 - fakeProb) * 100).toFixed(2)
    : (fakeProb * 100).toFixed(2);

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="surface p-6 space-y-5 text-center rounded-md"
    >
      <p className="text-md text-foreground">Prediction Result</p>

      <div
        className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium border ${config.bg} ${config.text} ${config.border}`}
      >
        {isValid ? <CloudCheck size={18} /> : <CloudAlert size={18} />}
        {config.label}
      </div>

      <p className="text-md text-foreground font-medium">
        {config.message}
      </p>

      <div className="space-y-2">
        <div className="flex justify-center gap-3 text-sm text-muted">
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