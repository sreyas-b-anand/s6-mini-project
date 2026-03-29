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
  const confidence = (result?.final_confidence * 100).toFixed(2);

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="surface p-6 space-y-5 text-center"
    >
      <p className="text-sm text-muted">Prediction Result</p>
      <div
        className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium ${
          isValid
            ? "bg-green-500/10 text-green-400 border border-green-500/20"
            : "bg-red-500/10 text-red-400 border border-red-500/20"
        }`}
      >
        {isValid ? <CloudCheck size={18} /> : <CloudAlert size={18} />}
        {isValid ? "Authentic Review" : "Fake Review"}
      </div>
      <div className="space-y-2">
        <div className="flex justify-between text-xs text-muted">
          <span>Confidence</span>
          <span className="text-foreground font-medium">{confidence}%</span>
        </div>

        <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidence}%` }}
            transition={{ duration: 0.6 }}
            className={`h-full ${isValid ? "bg-green-500" : "bg-red-500"}`}
          />
        </div>
      </div>
    </motion.div>
  );
};

export default MLResultComponent;
