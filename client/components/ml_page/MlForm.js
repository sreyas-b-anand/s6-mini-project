"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Loader } from "lucide-react";
import usePost from "@/hooks/usePost";
import MLResultComponent from "./MLResultComponent";

const MLForm = () => {
  const { postData, loading, data } = usePost("/ml_score");

  const [category, setCategory] = useState("");
  const [rating, setRating] = useState("");
  const [text, setText] = useState("");

  const handleSubmit = async () => {
    const formData = new FormData();

    formData.append("category", category);
    formData.append("rating", rating);
    formData.append("text", text);

    await postData(formData);
  };

  return (
    <div className="max-w-xl mx-auto space-y-8">
      <div className="text-center space-y-2">
        <p className="font-semibold text-3xl">Single Review Prediction</p>
        <p className="text-muted text-lg">
          Enter a review below to predict its authenticity
        </p>
      </div>

      <div className="space-y-6">
        <div className="flex justify-between items-center gap-4 md:flex-nowrap flex-wrap">
          <div className="space-y-2 flex-3/5">
            <Label className="text-base">Category</Label>
            <Select
              onValueChange={setCategory}
              className="focus-visible:ring-1 "
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select category" />
              </SelectTrigger>

              <SelectContent className="bg-background border">
                <SelectItem value="Electronics">Electronics</SelectItem>
                <SelectItem value="Books">Books</SelectItem>
                <SelectItem value="Movies">Movies & TV</SelectItem>
                <SelectItem value="Home Appliances">Home & Kitchen</SelectItem>
                <SelectItem value="Sports">Sports & Outdoors</SelectItem>
                <SelectItem value="Tools and Home Improvements">
                  Tools & Home Improvement
                </SelectItem>
                <SelectItem value="Pets supplies">Pet Supplies</SelectItem>
                <SelectItem value="Kindle">Kindle Store</SelectItem>
                <SelectItem value="Toys">Toys & Games</SelectItem>
                <SelectItem value="Fashion and clothing">
                  Fashion & Clothing
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2 flex-2/5">
            <Label className="text-base">Rating</Label>
            <Input
              type="number"
              min="1"
              max="5"
              value={rating}
              onChange={(e) => setRating(e.target.value)}
              placeholder="Enter rating (1-5)"
              className="focus-visible:ring-1 "
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label className="text-base">Review Text</Label>
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Write the review here..."
            className="min-h-30 focus-visible:ring-1 "
          />
        </div>

        <Button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full py-6 text-background hover:cursor-pointer hover:opacity-90"
        >
          {loading ? <Loader className="animate-spin" /> : "Predict Review using ML model"}
        </Button>
      </div>

      {<MLResultComponent result={data} />}
    </div>
  );
};

export default MLForm;
