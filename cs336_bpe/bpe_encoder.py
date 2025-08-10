# cs336_bpe/bpe_encoder.py
import regex as re
import pickle
from collections import defaultdict, Counter
import heapq

# Import GPT-2 mapping from tests instead of duplicating
# This ensures consistency with the test expectations
try:
    from tests.common import gpt2_bytes_to_unicode
except ImportError:
    # Fallback implementation if tests.common is not available
    def gpt2_bytes_to_unicode():
        """GPT-2 byte to unicode mapping (fallback implementation)"""
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        characters = [chr(n) for n in cs]
        d = dict(zip(bs, characters))
        return d

class BPETokenizer:
    """A Byte-Pair Encoding tokenizer implementation."""
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab: dict[int, bytes]
                Mapping from token ID to token bytes. Contains all tokens learned during training.
                Example: {
                    0: b'h',           # Single byte tokens (base vocabulary)
                    1: b'e', 
                    2: b'l',
                    3: b'o',
                    4: b' ',
                    5: b'll',          # Merged tokens from BPE training
                    6: b'ell',         # More complex merged tokens
                    7: b'hello',       # Even more complex merged tokens
                    ...
                }
                
            merges: list[tuple[bytes, bytes]]
                Ordered list of merge rules learned during BPE training. The order matters!
                Each tuple represents (token1, token2) that should be merged into token1+token2.
                Example: [
                    (b'l', b'l'),      # First merge: 'll' (most frequent pair)
                    (b'e', b'll'),     # Second merge: 'ell' 
                    (b'h', b'ell'),    # Third merge: 'hell'
                    (b'hell', b'o'),   # Fourth merge: 'hello'
                    (b' ', b'the'),    # Other common merges...
                    ...
                ]
                
            special_tokens: list[str] (optional)
                Tokens that should NEVER be split during encoding, regardless of BPE rules.
                These are treated as atomic units.
                Example: ["<|endoftext|>", "<|startoftext|>", "[UNK]", "[PAD]"]
        """
        # Store the vocabulary and create reverse lookup (bytes -> token_id)
        self.vocab = vocab                                    # token_id -> bytes
        self.reverse_vocab = {v: k for k, v in vocab.items()} # bytes -> token_id
        
        # Store merge rules in the order they were learned during training
        self.merges = merges
        
        # Store special tokens that should never be split
        self.special_tokens = special_tokens or []
        
        # Build merge priority map (currently unused, but useful for future optimizations)
        # Lower index = higher priority (earlier in training = more frequent)
        self.merge_priority = {merge: idx for idx, merge in enumerate(merges)}

    
    def encode(self, text: str) -> list[int]:
        """Convert text to list of token IDs."""
        # 1. Handle special tokens FIRST - split text around special tokens
        if self.special_tokens:
            text_segments = self._split_text_with_special_tokens(text)
        else:
            text_segments = [(text, False)]  # No special tokens, treat as single segment
        
        # 2. Process each segment
        all_token_ids = []
        for segment, is_special in text_segments:
            if is_special:
                # Special token: convert directly to token ID (no BPE applied)
                special_token_bytes = segment.encode('utf-8')
                if special_token_bytes in self.reverse_vocab:
                    all_token_ids.append(self.reverse_vocab[special_token_bytes])
                else:
                    raise ValueError(f"Special token '{segment}' not found in vocabulary")
            else:
                # Regular text: apply BPE encoding
                if segment:  # Skip empty segments
                    segment_token_ids = self._encode_regular_text(segment)
                    all_token_ids.extend(segment_token_ids)
        
        return all_token_ids
    
    def _apply_merge(self, tokens: list[bytes], token1: bytes, token2: bytes) -> list[bytes]:
        """Apply a single merge rule to the token sequence."""
        if len(tokens) < 2:
            return tokens
        
        # Note: Need to make sure "t1", "t2" are consecutive in the text, not just co-occurring
        merged_token = token1 + token2
        new_tokens = []
        i = 0
        
        while i < len(tokens):
            # Check if we can merge at current position
            if (i < len(tokens) - 1 and 
                tokens[i] == token1 and 
                tokens[i + 1] == token2):
                # Merge the pair
                new_tokens.append(merged_token)
                i += 2  # Skip both tokens
            else:
                # If cannot merge, Keep the current token
                new_tokens.append(tokens[i])
                i += 1
                
        return new_tokens
    
    def _split_text_with_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """
        Split text into segments, marking which are special tokens.
        
        Returns:
            List of (segment, is_special_token) tuples
            
        Example:
            Input: "Hello <|endoftext|> world"
            Output: [("Hello ", False), ("<|endoftext|>", True), (" world", False)]
        """
        if not self.special_tokens:
            return [(text, False)]
        
        # Sort special tokens by length (longest first) to handle overlapping cases
        special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        
        segments = []
        remaining_text = text
        
        while remaining_text:
            # Find the earliest occurring special token
            earliest_pos = len(remaining_text)
            earliest_token = None
            
            for special_token in special_tokens_sorted:
                pos = remaining_text.find(special_token)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_token = special_token
            
            if earliest_token is None:
                # No more special tokens found
                if remaining_text:
                    segments.append((remaining_text, False))
                break
            
            # Add text before special token (if any)
            if earliest_pos > 0:
                segments.append((remaining_text[:earliest_pos], False))
            
            # Add the special token
            segments.append((earliest_token, True))
            
            # Continue with text after special token
            remaining_text = remaining_text[earliest_pos + len(earliest_token):]
        
        return segments
    
    def _encode_regular_text(self, text: str) -> list[int]:
        """Apply BPE encoding to regular (non-special) text."""
        # 1. Convert text to initial tokens (individual bytes as bytes objects)
        tokens = [bytes([b]) for b in text.encode('utf-8')]  # Start with bytes objects
        
        # 2. Apply ALL merge rules in order
        for merge_rule in self.merges:
            token_1, token_2 = merge_rule
            tokens = self._apply_merge(tokens, token_1, token_2)
        
        # 3. Convert final tokens to token IDs using vocab
        token_ids = []
        for token in tokens:
            if token in self.reverse_vocab:
                token_ids.append(self.reverse_vocab[token])
            else:
                # Handle unknown tokens - this shouldn't happen with proper BPE
                raise ValueError(f"Token {token} not found in vocabulary")
        
        return token_ids
                
    
    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""
        decoded_bytes = b""
        for token_id in token_ids:
            if token_id in self.vocab:
                decoded_bytes += self.vocab[token_id]
            else:
                # Handle unknown token IDs - this shouldn't happen with proper BPE
                raise ValueError(f"Token ID {token_id} not found in vocabulary")
        
        # Convert bytes back to string
        return decoded_bytes.decode('utf-8', errors='replace')
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None):
        """Create a BPE tokenizer from files."""
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode_iterable(self, text_iterable):
        """Encode text from an iterable (like file lines)."""
        for line in text_iterable:
            line = line.rstrip('\n\r')  # Remove line endings
            if line:  # Skip empty lines
                for token_id in self.encode(line):
                    yield token_id


def get_tokenizer_implemented(vocab: dict[int, bytes], 
                  merges: list[tuple[bytes, bytes]], 
                  special_tokens: list[str] = None) -> BPETokenizer:
    """Factory function to create a BPE tokenizer."""
    return BPETokenizer(vocab, merges, special_tokens)

def get_tokenizer(vocab: dict[int, bytes], 
                  merges: list[tuple[bytes, bytes]], 
                  special_tokens: list[str] = None) -> BPETokenizer:
    """Factory function to create a BPE tokenizer."""
    return BPETokenizer(vocab, merges, special_tokens)

def run_train_bpe_implemented(input_path: str, vocab_size: int, special_tokens: list[str], use_advanced: bool = True, **kwargs):
    """Train a BPE tokenizer on a corpus.
    
    Args:
        input_path: Path to training corpus
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to preserve
        use_advanced: If True, use advanced optimizations (priority queue + incremental updates)
                     If False, use standard optimizations (word frequencies)
    """
    if use_advanced:
        return run_train_bpe_advanced(input_path, vocab_size, special_tokens, **kwargs)
    else:
        return run_train_bpe_standard(input_path, vocab_size, special_tokens, **kwargs)
        
def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], use_advanced: bool = True, **kwargs):
    """Train a BPE tokenizer on a corpus.
    
    Args:
        input_path: Path to training corpus
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to preserve
        use_advanced: If True, use advanced optimizations (priority queue + incremental updates)
                     If False, use standard optimizations (word frequencies)
    """
    if use_advanced:
        return run_train_bpe_advanced(input_path, vocab_size, special_tokens, **kwargs)
    else:
        return run_train_bpe_standard(input_path, vocab_size, special_tokens, **kwargs)
        
def run_train_bpe_standard(input_path: str, vocab_size: int, special_tokens: list[str], **kwargs):
    """Train a BPE tokenizer on a corpus (standard optimized version with GPT-2 encoding)."""
    from collections import defaultdict, Counter
    import heapq
    
    # Regex pattern for pre-tokenization (GPT-2 style)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Get GPT-2 byte-to-unicode mapping
    byte_encoder = gpt2_bytes_to_unicode()
    
    # Step 1: Initialize base vocabulary (all 256 bytes in GPT-2 encoding)
    vocab = {}
    for i in range(256):
        # Each token ID i should map to the raw byte i
        vocab[i] = bytes([i])
    
    # Step 2: Add special tokens to vocabulary
    if special_tokens:
        for special_token in special_tokens:
            special_token_bytes = special_token.encode('utf-8')
            if special_token_bytes not in vocab.values():
                vocab[len(vocab)] = special_token_bytes
    
    # Step 3: Pre-tokenize and collect word frequencies
    word_freqs = Counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Use regex to split line into tokens
            tokens = re.findall(PAT, line)
            for token in tokens:
                if token.strip():  # Skip empty tokens
                    # Convert each byte to GPT-2 unicode representation
                    gpt2_chars = []
                    for byte_val in token.encode('utf-8'):
                        gpt2_chars.append(byte_encoder[byte_val])
                    
                    word_tuple = tuple(gpt2_chars)
                    word_freqs[word_tuple] += 1
    
    # Convert to working format
    words = {word: list(word) for word in word_freqs.keys()}
    merges = []
    
    # Step 4: BPE training loop
    while len(vocab) < vocab_size:
        # Count pairs with frequency weighting
        pair_counts = defaultdict(int)
        for word_tuple, freq in word_freqs.items():
            word = words[word_tuple]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq  # Weight by word frequency
        
        if not pair_counts:
            break  # No more pairs to merge
        
        # Find most frequent pair
        most_frequent_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
        max_count = pair_counts[most_frequent_pair]
        
        if max_count == 0:
            break  # No pairs left
        
        # Create merged token (in GPT-2 unicode format)
        token1_str, token2_str = most_frequent_pair
        merged_token_str = token1_str + token2_str
        
        # Convert GPT-2 unicode strings back to raw bytes
        gpt2_byte_decoder = {v: k for k, v in byte_encoder.items()}
        token1_bytes = bytes([gpt2_byte_decoder[char] for char in token1_str])
        token2_bytes = bytes([gpt2_byte_decoder[char] for char in token2_str])
        merged_token_bytes = token1_bytes + token2_bytes
        
        # Add to vocabulary and merges
        vocab[len(vocab)] = merged_token_bytes
        merges.append((token1_bytes, token2_bytes))
        
        # Update all words by applying this merge
        new_word_freqs = Counter()
        for word_tuple, freq in word_freqs.items():
            word = words[word_tuple]
            new_word = _apply_merge_optimized_gpt2(word, token1_str, token2_str, merged_token_str)
            new_word_tuple = tuple(new_word)
            new_word_freqs[new_word_tuple] += freq
            words[new_word_tuple] = new_word
        
        word_freqs = new_word_freqs
        
        # Clean up old word entries to save memory
        words = {word: words.get(word, list(word)) for word in word_freqs.keys()}
    
    return vocab, merges

def _apply_merge_optimized(word: list, token1: bytes, token2: bytes, merged_token: bytes) -> list:
    """Apply a single merge operation efficiently."""
    if len(word) < 2:
        return word
    
    new_word = []
    i = 0
    while i < len(word):
        if (i < len(word) - 1 and 
            word[i] == token1 and 
            word[i + 1] == token2):
            new_word.append(merged_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word

def _apply_merge_optimized_gpt2(word: list, token1_str: str, token2_str: str, merged_token_str: str) -> list:
    """Apply a single merge operation efficiently for GPT-2 unicode strings."""
    if len(word) < 2:
        return word
    
    new_word = []
    i = 0
    while i < len(word):
        if (i < len(word) - 1 and 
            word[i] == token1_str and 
            word[i + 1] == token2_str):
            new_word.append(merged_token_str)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word

def run_train_bpe_advanced(input_path: str, vocab_size: int, special_tokens: list[str], **kwargs):
    """Train a BPE tokenizer with advanced optimizations (priority queue + incremental updates) using GPT-2 encoding."""
    from collections import defaultdict, Counter
    import heapq
    
    # Regex pattern for pre-tokenization (GPT-2 style)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Get GPT-2 byte-to-unicode mapping
    byte_encoder = gpt2_bytes_to_unicode()
    
    # Step 1: Initialize base vocabulary (all 256 bytes in GPT-2 encoding)
    vocab = {}
    for i in range(256):
        # Each token ID i should map to the raw byte i
        vocab[i] = bytes([i])
    
    # Step 2: Add special tokens to vocabulary
    if special_tokens:
        for special_token in special_tokens:
            special_token_bytes = special_token.encode('utf-8')
            if special_token_bytes not in vocab.values():
                vocab[len(vocab)] = special_token_bytes
    
    # Step 3: Pre-tokenize and collect word frequencies
    word_freqs = Counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = re.findall(PAT, line)
            for token in tokens:
                if token.strip():
                    # Convert each byte to GPT-2 unicode representation
                    gpt2_chars = []
                    for byte_val in token.encode('utf-8'):
                        gpt2_chars.append(byte_encoder[byte_val])
                    
                    word_tuple = tuple(gpt2_chars)
                    word_freqs[word_tuple] += 1
    
    # Convert to working format
    words = {word: list(word) for word in word_freqs.keys()}
    merges = []
    
    # ADVANCED OPTIMIZATION: Use priority queue with incremental updates
    def get_pair_counts():
        """Get initial pair counts."""
        pair_counts = defaultdict(int)
        for word_tuple, freq in word_freqs.items():
            word = words[word_tuple]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
        return pair_counts
    
    # Initial pair counting
    pair_counts = get_pair_counts()
    
    # Convert to max heap (negate values since heapq is min heap)
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    
    # Track which pairs are still valid
    valid_pairs = set(pair_counts.keys())
    
    while len(vocab) < vocab_size and heap:
        # Get the most frequent pair
        while heap:
            neg_count, pair = heapq.heappop(heap)
            count = -neg_count
            
            # Check if this pair is still valid (not outdated)
            if pair in valid_pairs and pair_counts[pair] == count:
                most_frequent_pair = pair
                max_count = count
                break
        else:
            break  # No valid pairs left
        
        if max_count == 0:
            break
        
        # Create merged token (in GPT-2 unicode format)
        token1_str, token2_str = most_frequent_pair
        merged_token_str = token1_str + token2_str
        
        # Convert GPT-2 unicode strings back to raw bytes
        gpt2_byte_decoder = {v: k for k, v in byte_encoder.items()}
        token1_bytes = bytes([gpt2_byte_decoder[char] for char in token1_str])
        token2_bytes = bytes([gpt2_byte_decoder[char] for char in token2_str])
        merged_token_bytes = token1_bytes + token2_bytes
        
        # Add to vocabulary and merges
        vocab[len(vocab)] = merged_token_bytes
        merges.append((token1_bytes, token2_bytes))
        
        # INCREMENTAL UPDATE: Only update affected words and pairs
        new_word_freqs = Counter()
        affected_pairs = set()
        
        for word_tuple, freq in word_freqs.items():
            word = words[word_tuple]
            
            # Check if this word contains the pair to merge
            has_pair = any(i < len(word) - 1 and 
                          word[i] == token1_str and word[i + 1] == token2_str 
                          for i in range(len(word) - 1))
            
            if has_pair:
                # Remove old pair counts for this word
                for i in range(len(word) - 1):
                    old_pair = (word[i], word[i + 1])
                    pair_counts[old_pair] -= freq
                    affected_pairs.add(old_pair)
                
                # Apply merge
                new_word = _apply_merge_optimized_gpt2(word, token1_str, token2_str, merged_token_str)
                new_word_tuple = tuple(new_word)
                
                # Add new pair counts for this word
                for i in range(len(new_word) - 1):
                    new_pair = (new_word[i], new_word[i + 1])
                    pair_counts[new_pair] += freq
                    affected_pairs.add(new_pair)
                    
                new_word_freqs[new_word_tuple] += freq
                words[new_word_tuple] = new_word
            else:
                # Word unchanged
                new_word_freqs[word_tuple] += freq
        
        word_freqs = new_word_freqs
        
        # Update heap with new/changed pair counts
        for pair in affected_pairs:
            count = pair_counts[pair]
            if count > 0:
                heapq.heappush(heap, (-count, pair))
                valid_pairs.add(pair)
            else:
                valid_pairs.discard(pair)
        
        # Remove the merged pair from valid pairs
        valid_pairs.discard(most_frequent_pair)
        
        # Clean up old word entries periodically to save memory
        if len(vocab) % 100 == 0:
            words = {word: words.get(word, list(word)) for word in word_freqs.keys()}
    
    return vocab, merges
