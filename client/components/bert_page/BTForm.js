"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Loader, AlertTriangle } from "lucide-react";
import { motion } from "framer-motion";
import usePost from "@/hooks/usePost";
import BTResultComponent from "./BTResultComponent";
import { BTMapComponent } from "./BTMapComponent";

const BTForm = () => {
  const { postData, loading, data } = usePost("/bert_score");

  const [review, setReview] = useState("");
  const [rating, setRating] = useState("");
  const [url, setUrl] = useState("");
  const [useLink, setUseLink] = useState(false);

  const handleSubmit = async () => {
    const formData = new FormData();

    if (useLink) {
      if (!url) return;
      formData.append("url", url);
      formData.append("type", "link");
    } else {
      if (!review || !rating) return;
      formData.append("review", review);
      formData.append("rating", rating);
      formData.append("type", "single");
    }

    await postData(formData);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="max-w-2xl mx-auto p-8 card-hover space-y-8 "
    >
      <div className="text-center space-y-3">
        <h1 className="text-3xl font-semibold">BERT Review Analysis</h1>
        <p className="text-sm text-muted">
          {useLink
            ? "Analyze multiple reviews using a product link"
            : "Analyze a single review using BERT"}
        </p>
      </div>

      <div className="space-y-6">
        <div className="flex justify-center">
          <div className="flex bg-secondary p-1 rounded-lg">
            <button
              onClick={() => setUseLink(false)}
              className={`px-4 py-1.5 text-sm rounded-md transition ${
                !useLink ? "bg-primary text-white" : "text-muted"
              }`}
            >
              Manual
            </button>
            <button
              onClick={() => setUseLink(true)}
              className={`px-4 py-1.5 text-sm rounded-md transition ${
                useLink ? "bg-primary text-white" : "text-muted"
              }`}
            >
              Link
            </button>
          </div>
        </div>
        {useLink && (
          <div className="flex gap-2 text-sm text-yellow-500 bg-yellow-500/10 border border-yellow-500/20 rounded-md p-3">
            <AlertTriangle size={16} />
            <p>Some reviews may require login on Amazon to be accessible</p>
          </div>
        )}

        {!useLink ? (
          <div className="space-y-4 md:w-103.5">
            <div className="space-y-2">
              <Label>Rating</Label>
              <Input
                type="number"
                value={rating}
                onChange={(e) => setRating(e.target.value)}
                placeholder="1 - 5"
                className="input-modern"
              />
            </div>

            <div className="space-y-2">
              <Label>Review</Label>
              <Textarea
                value={review}
                onChange={(e) => setReview(e.target.value)}
                placeholder="Write the review..."
                className="input-modern min-h-30"
              />
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            <Label>Product Link</Label>
            <Input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Paste Amazon product URL..."
              className="input-modern"
            />
          </div>
        )}

        <Button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full py-6 button-primary glow"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <Loader className="animate-spin" size={18} />
              Analyzing
            </span>
          ) : (
            "Analyze with BERT"
          )}
        </Button>
      </div>

      {!useLink && <BTResultComponent result={data} />}
      {useLink && <BTMapComponent reviews={data} />}
    </motion.div>
  );
};

export default BTForm;
