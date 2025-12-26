// Mock translation API service
// In a real implementation, this would connect to an actual translation API

class TranslationService {
  constructor() {
    this.cache = new Map();
    // Use environment variable if available, otherwise use default
    // In browser environment, process.env is not available, so we handle this gracefully
    this.apiEndpoint = typeof process !== 'undefined' ? (process.env?.TRANSLATION_API_URL || 'https://translate.googleapis.com/translate_a/single') : 'https://translate.googleapis.com/translate_a/single';
  }

  // Check if translation is cached
  getCachedTranslation(text, targetLang, sourceLang = 'en') {
    const cacheKey = this.generateCacheKey(text, targetLang, sourceLang);
    return this.cache.get(cacheKey);
  }

  // Cache a translation
  setCachedTranslation(text, targetLang, sourceLang = 'en', translation) {
    const cacheKey = this.generateCacheKey(text, targetLang, sourceLang);
    this.cache.set(cacheKey, {
      translation,
      timestamp: Date.now(),
      expiry: Date.now() + (24 * 60 * 60 * 1000) // 24 hours expiry
    });
  }

  // Generate cache key
  generateCacheKey(text, targetLang, sourceLang) {
    // Create a hash-like key from the inputs
    return `${text.substring(0, 50)}_${targetLang}_${sourceLang}`;
  }

  // Clean expired cache entries
  cleanExpiredCache() {
    const now = Date.now();
    for (const [key, value] of this.cache.entries()) {
      if (now > value.expiry) {
        this.cache.delete(key);
      }
    }
  }

  // Translate text using mock service (replace with real API in production)
  async translateText(text, targetLang, sourceLang = 'en') {
    // Clean expired cache entries
    this.cleanExpiredCache();

    // Check cache first
    const cached = this.getCachedTranslation(text, targetLang, sourceLang);
    if (cached && Date.now() < cached.expiry) {
      return cached.translation;
    }

    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 500));

    // Mock translation - in production, replace with actual API call
    let translatedText = text;

    if (targetLang === 'ur') {
      // Mock Urdu translation - in real implementation, use actual translation API
      // This is just for demonstration purposes
      translatedText = this.mockUrduTranslation(text);
    } else if (targetLang === 'en' && sourceLang === 'ur') {
      // Mock reverse translation
      translatedText = text.replace(/\[URDU\] (.+?) \[ENGLISH\]/g, '$1');
    }

    // Cache the result
    this.setCachedTranslation(text, targetLang, sourceLang, translatedText);

    return translatedText;
  }

  // Mock Urdu translation function (for demonstration)
  mockUrduTranslation(text) {
    // This is a placeholder - in a real implementation, you would use an actual translation API
    // For now, we'll just add a marker to show that translation occurred
    const sentences = text.split(/([.!?]+)/);
    let translated = '';

    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i];
      if (sentence.trim() && !['.', '!', '?', '. ', '! ', '? '].includes(sentence.trim())) {
        // Add Urdu translation marker
        translated += `[URDU] ${sentence.trim()} [ENGLISH]`;
      } else {
        translated += sentence;
      }
    }

    return translated;
  }

  // Translate multiple texts
  async translateMultiple(texts, targetLang, sourceLang = 'en') {
    const results = [];
    for (const text of texts) {
      const translation = await this.translateText(text, targetLang, sourceLang);
      results.push(translation);
    }
    return results;
  }

  // Get supported languages
  getSupportedLanguages() {
    return [
      { code: 'en', name: 'English' },
      { code: 'ur', name: 'Urdu' }
    ];
  }

  // Check if language is supported
  isLanguageSupported(langCode) {
    return this.getSupportedLanguages().some(lang => lang.code === langCode);
  }
}

// Create a singleton instance
const translationService = new TranslationService();
export default translationService;