import { useState } from "react";
import axiosInstance from "@/config/axios";

const usePost = (url) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const postData = async (body) => {
    try {
      setLoading(true);
      setError(null);

      const res = await axiosInstance.post(url, body);

      if (!res.data) {
        setError("No data received from the server");
        return;
      }

      setData(res.data);
      console.log(res.data);
      return res.data;
    } catch (err) {
      setError(err);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { postData, data, loading, error };
};

export default usePost;
