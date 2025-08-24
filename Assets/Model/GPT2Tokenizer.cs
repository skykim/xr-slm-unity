using UnityEngine;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

public class GPT2Tokenizer : MonoBehaviour
{
    public TextAsset vocabFile;
    public TextAsset mergesFile;
    public TextAsset tokenizerConfigFile;

    private Dictionary<string, int> _encoder;
    private Dictionary<int, string> _decoder;
    private Dictionary<byte, char> _byteEncoder;
    private Dictionary<char, byte> _byteDecoder;
    private Dictionary<(string, string), int> _bpeRanks;
    private readonly Dictionary<string, List<string>> _bpeCache = new Dictionary<string, List<string>>();
    
    private HashSet<string> _specialTokens;
    private Regex _specialTokensRegex;
    private Regex _pretokenizeRegex;
    
    public int EosTokenId { get; private set; }
    public int PadTokenId { get; private set; }
    public int UnkTokenId { get; private set; }
    
    private bool _isInitialized = false;

    private const string PretokenizePattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    void Awake()
    {
        Initialize();
    }
    
    public void Initialize()
    {
        if (_isInitialized) return;

        if (vocabFile == null || mergesFile == null || tokenizerConfigFile == null)
        {
            Debug.LogError("Tokenizer 파일 3개(vocab, merges, config)가 모두 할당되어야 합니다.");
            return;
        }

        _encoder = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabFile.text);
        
        var tokenizerConfig = JsonConvert.DeserializeObject<TokenizerConfig>(tokenizerConfigFile.text);

        if (tokenizerConfig?.AddedTokensDecoder != null)
        {
            foreach (var kvp in tokenizerConfig.AddedTokensDecoder)
            {
                if (int.TryParse(kvp.Key, out int tokenId))
                {
                    string tokenContent = kvp.Value.Content;
                    _encoder[tokenContent] = tokenId;
                }
            }
        }
        _decoder = _encoder.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        
        _specialTokens = new HashSet<string>();
        if (tokenizerConfig?.AddedTokensDecoder != null)
        {
            foreach (var tokenDef in tokenizerConfig.AddedTokensDecoder.Values)
            {
                if (tokenDef.Special)
                {
                    _specialTokens.Add(tokenDef.Content);
                }
            }
        }
        
        if (_specialTokens.Any())
        {
            var escapedTokens = _specialTokens.Select(Regex.Escape);
            _specialTokensRegex = new Regex($"({string.Join("|", escapedTokens)})", RegexOptions.Compiled);
        }
        else
        {
            _specialTokensRegex = new Regex("(?!)", RegexOptions.Compiled);
        }

        EosTokenId = _encoder[tokenizerConfig.EosToken];
        PadTokenId = _encoder[tokenizerConfig.PadToken];
        UnkTokenId = _encoder[tokenizerConfig.UnkToken];
        
        _bpeRanks = LoadMergesFromString(mergesFile.text);

        (_byteEncoder, _byteDecoder) = BuildByteToUnicodeMap();
        _pretokenizeRegex = new Regex(PretokenizePattern, RegexOptions.Compiled);

        _isInitialized = true;
        Debug.Log("GPT-2 Tokenizer (Advanced)가 성공적으로 초기화되었습니다.");
    }

    public List<int> Encode(string text)
    {
        if (!_isInitialized)
        {
            Debug.LogError("Tokenizer가 초기화되지 않았습니다.");
            return new List<int>();
        }

        text = text.Normalize(NormalizationForm.FormC);
        var tokenIds = new List<int>();

        string[] parts = _specialTokensRegex.Split(text);
        
        foreach (string part in parts)
        {
            if (string.IsNullOrEmpty(part)) continue;

            if (_specialTokens.Contains(part))
            {
                tokenIds.Add(_encoder[part]);
            }
            else
            {
                var matches = _pretokenizeRegex.Matches(part);
                foreach (Match match in matches)
                {
                    var builder = new StringBuilder();
                    foreach (byte b in Encoding.UTF8.GetBytes(match.Value))
                    {
                        builder.Append(_byteEncoder[b]);
                    }
                    
                    List<string> bpeTokens = Bpe(builder.ToString());
                    foreach (string token in bpeTokens)
                    {
                        if (_encoder.TryGetValue(token, out int id))
                        {
                            tokenIds.Add(id);
                        }
                        else
                        {
                            Debug.LogWarning($"'{token}' not found in vocab. Using UNK token.");
                            tokenIds.Add(UnkTokenId);
                        }
                    }
                }
            }
        }
        return tokenIds;
    }

    public string Decode(List<int> tokenIds)
    {
        if (!_isInitialized) return string.Empty;

        var builder = new StringBuilder();
        foreach (int id in tokenIds)
        {
            if (_decoder.TryGetValue(id, out string token))
            {
                if (_specialTokens.Contains(token))
                {
                    builder.Append(token);
                }
                else
                {
                    var byteBuffer = new List<byte>();
                    foreach (char c in token)
                    {
                        if (_byteDecoder.TryGetValue(c, out byte b))
                        {
                            byteBuffer.Add(b);
                        }
                    }
                    builder.Append(Encoding.UTF8.GetString(byteBuffer.ToArray()));
                }
            }
        }
        
        return builder.ToString();
    }

    private List<string> Bpe(string token)
    {
        if (_bpeCache.TryGetValue(token, out var cachedResult)) return cachedResult;
        if (token.Length <= 1)
        {
            var result = new List<string> { token };
            _bpeCache[token] = result;
            return result;
        }

        var word = token.Select(c => c.ToString()).ToList();

        while (word.Count > 1)
        {
            var pairs = GetPairs(word);
            var bestPair = pairs.OrderBy(p => _bpeRanks.GetValueOrDefault(p, int.MaxValue)).First();
            if (!_bpeRanks.ContainsKey(bestPair)) break;
            
            var newWord = new List<string>();
            int i = 0;
            while (i < word.Count)
            {
                if (i < word.Count - 1 && word[i] == bestPair.Item1 && word[i + 1] == bestPair.Item2)
                {
                    newWord.Add(bestPair.Item1 + bestPair.Item2);
                    i += 2;
                }
                else
                {
                    newWord.Add(word[i]);
                    i++;
                }
            }
            word = newWord;
        }
        
        _bpeCache[token] = word;
        return word;
    }

    private static HashSet<(string, string)> GetPairs(List<string> word)
    {
        var pairs = new HashSet<(string, string)>();
        if (word.Count < 2) return pairs;
        for (int i = 0; i < word.Count - 1; i++)
        {
            pairs.Add((word[i], word[i + 1]));
        }
        return pairs;
    }

    private static Dictionary<(string, string), int> LoadMergesFromString(string mergesContent)
    {
        var ranks = new Dictionary<(string, string), int>();
        var lines = mergesContent.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        int rank = 0;
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
            var parts = line.Split(' ');
            if (parts.Length == 2) ranks[(parts[0], parts[1])] = rank++;
        }
        return ranks;
    }
    
    private static (Dictionary<byte, char>, Dictionary<char, byte>) BuildByteToUnicodeMap()
    {
        var byteToUnicode = new Dictionary<byte, char>();
        var unicodeToByte = new Dictionary<char, byte>();
        
        var visibleChars = Enumerable.Range('!', '~' - '!' + 1)
            .Concat(Enumerable.Range('¡', '¬' - '¡' + 1))
            .Concat(Enumerable.Range('®', 'ÿ' - '®' + 1))
            .Select(i => (byte)i).ToHashSet();

        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            char mappedChar;
            if (visibleChars.Contains((byte)b))
            {
                mappedChar = (char)b;
            }
            else
            {
                mappedChar = (char)(256 + n);
                n++;
            }
            byteToUnicode[(byte)b] = mappedChar;
            unicodeToByte[mappedChar] = (byte)b;
        }
        return (byteToUnicode, unicodeToByte);
    }

    private class TokenizerConfig
    {
        [JsonProperty("eos_token")]
        public string EosToken { get; set; }

        [JsonProperty("pad_token")]
        public string PadToken { get; set; }

        [JsonProperty("unk_token")]
        public string UnkToken { get; set; }

        [JsonProperty("added_tokens_decoder")]
        public Dictionary<string, AddedTokenDef> AddedTokensDecoder { get; set; }
    }

    private class AddedTokenDef
    {
        [JsonProperty("content")]
        public string Content { get; set; }
        
        [JsonProperty("special")]
        public bool Special { get; set; }
    }
}