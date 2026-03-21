"use client";

import React, { use, useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Loader } from "lucide-react";
import usePost from "@/hooks/usePost";
import BTResultComponent from "./BTResultComponent";
import { form } from "framer-motion/client";
import { number } from "framer-motion";
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
      formData.append("link", url);
      formData.append("type", "link");
    } else {
      formData.append("review", review);
      formData.append("rating", rating);
      formData.append("type", "single");
    }

    await postData(formData);
  };

  return (
    <div className="max-w-xl mx-auto space-y-8">
      <div className="text-center space-y-2">
        <p className="font-semibold text-3xl">BERT Review Prediction</p>
        <p className="text-muted text-md">
          Enter a review or a product link below to predict its authenticity
        </p>
      </div>

      <div className="space-y-6">
        <div className="flex justify-end">
          <Button
            onClick={() => setUseLink(!useLink)}
            className="text-sm text-gray-100 focus:outline-none  hover:cursor-pointer hover:opacity-90"
          >
            {useLink ? "Enter reviews manually" : "Use product link"}
          </Button>
        </div>

        {!useLink && (
          <div className="space-y-2">
            <Label className="text-base">Rating</Label>
            <Input
              type={number}
              value={rating}
              onChange={(e) => setRating(e.target.value)}
              placeholder="Enter the rating..."
              className="focus-visible:ring-1"
            />
            <Label className="text-base">Review Text</Label>
            <Textarea
              value={review}
              onChange={(e) => setReview(e.target.value)}
              placeholder="Write the review here..."
              className="min-h-40 focus-visible:ring-1"
            />
          </div>
        )}

        {/* Product Link Input */}
        {useLink && (
          <div className="space-y-2">
            <Label className="text-base">Product Link</Label>
            <Input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Paste the product URL here..."
              className="focus-visible:ring-1"
            />
          </div>
        )}

        <Button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full py-6 text-background hover:cursor-pointer hover:opacity-90"
        >
          {loading ? (
            <Loader className="animate-spin" />
          ) : (
            "Predict Review using BERT"
          )}
        </Button>
      </div>

      {!useLink && <BTResultComponent result={data} />}
      
      {useLink && <BTMapComponent reviews={data} />}
    </div>
  );
};

export default BTForm;
