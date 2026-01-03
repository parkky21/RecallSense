# Image Retrieval Models - Performance Results

This document contains performance metrics and search results for the three embedding models used in the image retrieval system.

## Models Overview

1. **Qwen/Qwen3-Embedding-0.6B** - Qwen embedding model (600M parameters)
2. **Alibaba-NLP/gte-multilingual-base** - GTE-multilingual embedding model
3. **google/embeddinggemma-300m** - Google EmbeddingGemma model (300M parameters)

---

## Indexing Performance

### Test Dataset
- **Total Images**: 17 images
- **Device**: CPU (CUDA not available)
- **Date**: Test run from terminal output

### Embedding Generation Times

| Model | Total Time | Per Image Time | Speed Ranking |
|-------|------------|----------------|---------------|
| **GTE-multilingual** | 1.72 seconds | 0.1009 seconds | 🥇 Fastest |
| **EmbeddingGemma** | 2.38 seconds | 0.1400 seconds | 🥈 Second |
| **Qwen** | 7.37 seconds | 0.4335 seconds | 🥉 Slowest |

### Performance Analysis
- **GTE-multilingual** is the fastest, taking only 1.72 seconds for 17 images (0.10s per image)
- **EmbeddingGemma** is moderately fast, taking 2.38 seconds (0.14s per image)
- **Qwen** is the slowest, taking 7.37 seconds (0.43s per image) - approximately 4.3x slower than GTE

---

## Search Performance

### Test Query: "bike"

### Search Timing (for 17 indexed images)

| Model | Search Time | Speed Ranking |
|-------|-------------|---------------|
| **GTE-multilingual** | 0.3971 seconds | 🥇 Fastest |
| **EmbeddingGemma** | 0.9877 seconds | 🥈 Second |
| **Qwen** | 2.9163 seconds | 🥉 Slowest |

### Performance Analysis
- **GTE-multilingual** is fastest for search (0.40s)
- **EmbeddingGemma** is moderate (0.99s)
- **Qwen** is slowest (2.92s) - approximately 7.3x slower than GTE

---

## Search Results Comparison

### Query: "bike"

#### Qwen Model Results

| Rank | Similarity | Image | Caption |
|------|------------|-------|---------|
| 1 | 0.3831 | Screenshot 2025-12-04 132720.png | a man is riding a motorcycle on the street |
| 2 | 0.2454 | wallpaper.png | a tori tori floating in the water |
| 3 | 0.2336 | 1314707.jpg | a girl with long hair and a bow on her head |

**Analysis**: 
- ✅ Correctly identified the motorcycle image as top result
- ⚠️ Lower similarity scores overall (max 0.38)
- ⚠️ Second result (tori floating in water) seems unrelated

#### GTE-multilingual Model Results

| Rank | Similarity | Image | Caption |
|------|------------|-------|---------|
| 1 | 0.5844 | Screenshot 2025-12-04 132720.png | a man is riding a motorcycle on the street |
| 2 | 0.5578 | wallpaper.png | a tori tori floating in the water |
| 3 | 0.5199 | Screenshot 2025-12-03 122250.png | a tray with two cups of food and a drink |

**Analysis**:
- ✅ Correctly identified the motorcycle image as top result
- ✅ Higher similarity scores (max 0.58) - better confidence
- ⚠️ Second result still seems unrelated (tori floating in water)
- ✅ Third result different from Qwen (food tray vs girl with bow)

#### EmbeddingGemma Model Results

| Rank | Similarity | Image | Caption |
|------|------------|-------|---------|
| 1 | 0.3735 | Screenshot 2025-12-04 132720.png | a man is riding a motorcycle on the street |
| 2 | 0.2694 | ultra-instinct-goku-dragon-ball-super-5k-5760x3240-5127.jpg | dragon ball wallpaper by the - dragon - ballpaper |
| 3 | 0.2616 | Screenshot 2025-12-04 132823.png | a woman standing next to a tree |

**Analysis**:
- ✅ Correctly identified the motorcycle image as top result
- ⚠️ Similarity scores similar to Qwen (max 0.37)
- ⚠️ Different second and third results compared to other models
- ⚠️ Results seem less relevant (Dragon Ball wallpaper, woman by tree)

---

## Model Comparison Summary

### Speed Performance
1. **GTE-multilingual**: Fastest for both indexing and search
2. **EmbeddingGemma**: Moderate speed
3. **Qwen**: Slowest but still functional

### Accuracy Performance (for "bike" query)
1. **GTE-multilingual**: 
   - Highest similarity scores (0.58 for correct result)
   - Best confidence in results
   - Fastest search time
   
2. **Qwen**: 
   - Moderate similarity scores (0.38 for correct result)
   - Correctly identified top result
   - Slower but accurate
   
3. **EmbeddingGemma**: 
   - Similar similarity scores to Qwen (0.37 for correct result)
   - Correctly identified top result
   - Different ranking for lower results

### Recommendations

**For Speed-Critical Applications:**
- Use **GTE-multilingual** - fastest overall performance

**For Accuracy-Critical Applications:**
- Use **GTE-multilingual** - highest similarity scores and best confidence
- Consider **Qwen** as alternative - good accuracy but slower

**For Multilingual Support:**
- **GTE-multilingual** - explicitly designed for multilingual tasks
- **EmbeddingGemma** - may have multilingual capabilities

**For Balanced Performance:**
- **GTE-multilingual** - best balance of speed and accuracy

---

## Notes

- All models correctly identified the motorcycle image as the most relevant result for the "bike" query
- Similarity scores vary significantly between models (GTE shows higher confidence)
- Search results show some variation in ranking, especially for lower-ranked items
- Performance was tested on CPU; GPU acceleration would improve all models' speed
- Results may vary based on query type, image content, and caption quality

---

## Future Testing Recommendations

1. Test with more diverse queries (objects, actions, scenes, abstract concepts)
2. Test with larger image datasets (100+, 1000+ images)
3. Compare results on GPU vs CPU
4. Evaluate multilingual query performance
5. Test with different image types (photos, illustrations, screenshots, etc.)
6. Measure precision@k and recall@k metrics
7. User preference studies for result quality

---

*Last Updated: Based on test run from terminal output*
*Test Environment: Windows, CPU-only, 17 images*

