import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
import tiktoken
import random
import torch.nn as nn
from words import words

def find_best_word_to_add(
    model: AutoModelForSequenceClassification, 
    tokenizer: AutoTokenizer, 
    injection_text: str, 
    adv_prefix: str, 
    text: str, 
    benign_class_idx: int, 
    device: torch.device,
    num_candidates: int = 20
) -> Tuple[Optional[str], float]:
    """
    Evaluate multiple candidate words and find the one that most improves the benign score when added to the prefix.
    
    Parameters:
    -----------
    model: The model to evaluate with
    tokenizer: The tokenizer to use
    injection_text: The injection text to prepend
    adv_prefix: The current prefix
    text: The text to append after the prefix
    benign_class_idx: The index of the benign class
    num_candidates: Number of candidate words to test
    
    Returns:
    --------
    best_word: The word that most improves the benign score
    improvement: The amount of improvement in benign score
    """
    print(f"\n----- TESTING {num_candidates} CANDIDATE WORDS TO ADD -----")
    
    # Get baseline benign score with current prefix
    try:
        inputs: Dict[str, torch.Tensor] = tokenizer(injection_text + adv_prefix + text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits: torch.Tensor = model(**inputs).logits
            probs: torch.Tensor = torch.softmax(logits, dim=-1)
            baseline_score: float = probs[0][benign_class_idx].item()
        print(f"Baseline benign score: {baseline_score:.4f}")
    except Exception as e:
        print(f"Error testing baseline: {e}")
        return None, 0
    
    # Generate candidate words to test
    # Use a mix of different types of words for diversity
    candidate_pools: List[List[str]] = [
        random.choices(words, k=num_candidates),
    ]
    
    candidates: List[str] = []
    for pool in candidate_pools:
        candidates.extend(pool)
    
    # Ensure we have the right number of candidates
    if len(candidates) > num_candidates:
        candidates = random.sample(candidates, num_candidates)
    elif len(candidates) < num_candidates:
        additional: List[str] = random.choices(words, k=num_candidates - len(candidates))
        candidates.extend(additional)
    
    # Test each candidate word
    best_word: Optional[str] = None
    best_score: float = baseline_score
    best_improvement: float = 0
    
    insert_positions: List[str] = ["beginning", "middle", "end"]
    results: List[Dict[str, Any]] = []
    
    for word in candidates:
        for position in insert_positions:
            # Create test prefix with the candidate word
            if position == "beginning":
                test_prefix: str = word + " " + adv_prefix
            elif position == "end":
                test_prefix = adv_prefix + " " + word
            else:  # middle
                # Find a reasonable spot to insert in the middle if possible
                if ' ' in adv_prefix:
                    words_list: List[str] = adv_prefix.split()
                    middle_idx: int = len(words_list) // 2
                    words_list.insert(middle_idx, word)
                    test_prefix = ' '.join(words_list)
                else:
                    # If no spaces, insert at midpoint of string
                    middle_idx: int = len(adv_prefix) // 2
                    test_prefix = adv_prefix[:middle_idx] + " " + word + " " + adv_prefix[middle_idx:]
            
            try:
                inputs: Dict[str, torch.Tensor] = tokenizer(injection_text + test_prefix + text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits: torch.Tensor = model(**inputs).logits
                    probs: torch.Tensor = torch.softmax(logits, dim=-1)
                    benign_score: float = probs[0][benign_class_idx].item()
                
                improvement: float = benign_score - baseline_score
                token_count: int = len(tokenizer.encode(test_prefix, add_special_tokens=False))
                
                results.append({
                    "word": word,
                    "position": position,
                    "score": benign_score,
                    "improvement": improvement,
                    "tokens": token_count,
                    "prefix": test_prefix
                })
                
                print(f"Word '{word}' at {position}: {benign_score:.4f} (Δ: {improvement:.4f}, tokens: {token_count})")
                
                if benign_score > best_score:
                    best_score = benign_score
                    best_word = word
                    best_improvement = improvement
                    best_position: str = position
                    best_prefix: str = test_prefix
            except Exception as e:
                print(f"Error testing word '{word}' at {position}: {e}")
                continue
    
    # Sort results by improvement
    results.sort(key=lambda x: x["improvement"], reverse=True)
    
    # Print top 5 results
    print("\nTop 5 most effective additions:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. '{result['word']}' at {result['position']}: {result['score']:.4f} (Δ: {result['improvement']:.4f}, tokens: {result['tokens']})")
    
    if best_word:
        print(f"\nBest word to add: '{best_word}' at {best_position}")
        print(f"Improvement: {best_improvement:.4f} (from {baseline_score:.4f} to {best_score:.4f})")
        print(f"New prefix: {best_prefix}")
        return best_prefix, best_improvement
    else:
        print("No improvement found from any candidate word")
        return None, 0


def token_gradients_combined(
    model: AutoModelForSequenceClassification, 
    input_ids: torch.Tensor, 
    input_slice: slice, 
    device: torch.device,
    benign_class: int = 1, 
    malicious_class: int = 0, 
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Computes gradients using a combination of loss minimization and benign maximization.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    benign_class : int
        The benign class index to optimize for.
    malicious_class : int
        The malicious class index to minimize.
    alpha : float
        Weight for the benign optimization (1-alpha is weight for loss minimization)

    Returns
    -------
    torch.Tensor
        The combined gradients for token optimization.
    """

    embed_weights: torch.Tensor = model.deberta.embeddings.word_embeddings.weight
    one_hot: torch.Tensor = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds: torch.Tensor = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds: torch.Tensor = model.deberta.embeddings.word_embeddings(input_ids)
    full_embeds: torch.Tensor = torch.cat(
        [
            embeds[:input_slice.start,:],
            input_embeds.squeeze(),
            embeds[input_slice.stop:,:]
        ],
        dim=0)
    logits: torch.Tensor = model(inputs_embeds=full_embeds.unsqueeze(0)).logits

    # Combined loss: minimize malicious class (standard loss) and maximize benign class
    standard_loss: torch.Tensor = nn.CrossEntropyLoss()(logits, torch.zeros(logits.shape[0], device=device).long())

    # Maximize benign class probability
    log_probs: torch.Tensor = torch.log_softmax(logits, dim=1)
    benign_loss: torch.Tensor = -log_probs[0, benign_class]

    # Combined loss with weighting
    combined_loss: torch.Tensor = (1 - alpha) * standard_loss + alpha * benign_loss
    combined_loss.backward()

    return one_hot.grad.clone()


def analyze_token_contributions(
    model: AutoModelForSequenceClassification, 
    tokenizer: AutoTokenizer, 
    injection_text: str, 
    adv_prefix: str, 
    text: str, 
    benign_class_idx: int, 
    min_benign_confidence: float, 
    device: torch.device,
    min_acceptable_benign: float = 0.6,
) -> str:
    """
    Analyze which tokens contribute most to the benign rating and systematically remove the least important ones.
    
    This performs an ablation study on the tokens in the prefix and iteratively removes tokens
    that contribute the least to maintaining the benign classification.
    """
    print("\n===== STARTING TOKEN CONTRIBUTION ANALYSIS =====")
    
    # Check original prefix
    prefix_token_ids: torch.Tensor = tokenizer.encode(adv_prefix, add_special_tokens=False)
    original_token_count: int = len(prefix_token_ids)
    prefix_tokens: List[str] = tokenizer.convert_ids_to_tokens(prefix_token_ids)
    
    print(f"Original prefix: '{adv_prefix}'")
    print(f"Original token count: {original_token_count}")
    print(f"Token breakdown: {prefix_tokens}")
    
    # Get original benign score
    try:
        inputs: Dict[str, torch.Tensor] = tokenizer(injection_text + adv_prefix + text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits: torch.Tensor = model(**inputs).logits
            probs: torch.Tensor = torch.softmax(logits, dim=-1)
            original_benign_score: float = probs[0][benign_class_idx].item()
        print(f"Original benign score: {original_benign_score:.4f}")
    except Exception as e:
        print(f"Error testing original prefix: {e}")
        return adv_prefix
    
    # If we don't meet the minimum threshold, adjust it
    if original_benign_score < min_acceptable_benign:
        min_acceptable_benign = original_benign_score * 0.95
        print(f"Adjusted minimum acceptable threshold to {min_acceptable_benign:.4f}")
    
    best_prefix: str = adv_prefix
    current_prefix: str = adv_prefix
    current_token_ids: List[int] = prefix_token_ids.copy()
    current_benign_score: float = original_benign_score
    
    print("\n----- ITERATIVE TOKEN ABLATION -----")
    
    # Keep removing tokens until we can't remove any more
    iteration: int = 0
    while len(current_token_ids) > 1:
        iteration += 1
        print(f"\nIteration {iteration}: Testing removal of individual tokens")
        print(f"Current token count: {len(current_token_ids)}")
        print(f"Current tokens: {tokenizer.convert_ids_to_tokens(current_token_ids)}")
        print(f"Current benign score: {current_benign_score:.4f}")
        
        best_removal_idx: Optional[int] = None
        best_removal_score: float = -1
        best_removal_prefix: Optional[str] = None
        
        # Test removing each token
        for i in range(len(current_token_ids)):
            # Create a version without this token
            test_token_ids: List[int] = current_token_ids.copy()
            removed_token_id: int = test_token_ids.pop(i)
            removed_token: str = tokenizer.convert_ids_to_tokens([removed_token_id])[0]
            
            # Skip if empty
            if not test_token_ids:
                continue
                
            test_prefix: str = tokenizer.decode(test_token_ids)
            
            # Skip if empty after decoding
            if not test_prefix.strip() and current_benign_score >= min_acceptable_benign:
                continue
            
            try:
                inputs: Dict[str, torch.Tensor] = tokenizer(injection_text + test_prefix + text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits: torch.Tensor = model(**inputs).logits
                    probs: torch.Tensor = torch.softmax(logits, dim=-1)
                    benign_score: float = probs[0][benign_class_idx].item()
                
                print(f"  Removing token {i} '{removed_token}': benign_score={benign_score:.4f}")
                
                # Is this the best removal so far?
                if benign_score >= min_acceptable_benign and benign_score > best_removal_score:
                    best_removal_idx = i
                    best_removal_score = benign_score
                    best_removal_prefix = test_prefix
            except Exception as e:
                print(f"  Error testing removal of token {i}: {e}")
                continue
        
        # If we found a token to remove that keeps acceptable classification
        if best_removal_idx is not None:
            removed_token_id: int = current_token_ids.pop(best_removal_idx)
            removed_token: str = tokenizer.convert_ids_to_tokens([removed_token_id])[0]
            current_prefix = best_removal_prefix
            current_benign_score = best_removal_score
            
            print(f"SUCCESS! Removed token {best_removal_idx} '{removed_token}', new benign score: {current_benign_score:.4f}")
            print(f"New prefix: '{current_prefix}'")
            print(f"New token count: {len(current_token_ids)}")
            
            # Update best prefix if it's better than our current best
            if best_removal_score >= min_acceptable_benign:
                best_prefix = current_prefix
        else:
            # If we can't remove any more tokens without dropping below threshold
            print("Cannot remove any more tokens while maintaining acceptable benign score.")
            break
    
    # Try a few final options if we ended up with a very short prefix
    if len(current_token_ids) <= 3:
        print("\n----- TESTING ADDITIONAL MINIMAL OPTIONS -----")
        
        for test_prefix in ["A", "The", "Safe", "Benign", "Educational", "Academic", ".", " ", "✓"]:
            try:
                inputs: Dict[str, torch.Tensor] = tokenizer(injection_text + test_prefix + text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits: torch.Tensor = model(**inputs).logits
                    probs: torch.Tensor = torch.softmax(logits, dim=-1)
                    benign_score: float = probs[0][benign_class_idx].item()
                
                test_token_count: int = len(tokenizer.encode(test_prefix, add_special_tokens=False))
                print(f"Option '{test_prefix}': tokens={test_token_count}, benign_score={benign_score:.4f}")
                
                # Is this a better option?
                if benign_score >= min_acceptable_benign and test_token_count < len(current_token_ids):
                    best_prefix = test_prefix
                    current_prefix = test_prefix
                    current_benign_score = benign_score
                    current_token_ids = tokenizer.encode(test_prefix, add_special_tokens=False)
                    print(f"SUCCESS! Found better minimal prefix: '{test_prefix}' with {test_token_count} tokens")
            except Exception as e:
                continue
    
    # Report results
    final_token_count: int = len(tokenizer.encode(best_prefix, add_special_tokens=False))
    reduction: float = ((original_token_count - final_token_count) / original_token_count * 100) if original_token_count > 0 else 0
    
    print("\n===== TOKEN ABLATION COMPLETE =====")
    print(f"Original prefix: '{adv_prefix}'")
    print(f"Original token count: {original_token_count}")
    print(f"Original benign score: {original_benign_score:.4f}")
    print(f"Final prefix: '{best_prefix}'")
    print(f"Final token count: {final_token_count}")
    print(f"Final benign score: {current_benign_score:.4f}")
    print(f"Reduction: {reduction:.2f}%")
    
    return best_prefix

def minimize_tokens(
    model: AutoModelForSequenceClassification, 
    tokenizer: AutoTokenizer, 
    injection_text: str, 
    adv_prefix: str, 
    text: str, 
    benign_class_idx: int, 
    min_benign_confidence: float, 
    device: torch.device,
    target_tokens: int = 1, 
    min_acceptable_benign: float = 0.6
) -> str:
    """
    Minimize tokens using only token contribution analysis (ablation study).
    This approach systematically removes tokens that contribute least to the benign classification.
    """
    print("\n===== STARTING TOKEN MINIMIZATION =====")
    
    # Use only token ablation approach - systematically remove tokens that contribute least
    ablation_prefix: str = analyze_token_contributions(
        model, tokenizer, injection_text, adv_prefix, text, 
        benign_class_idx, min_benign_confidence=min_benign_confidence,
        device=device, min_acceptable_benign=min_acceptable_benign
    )
    
    # Report final token count
    final_token_count: int = len(tokenizer.encode(ablation_prefix, add_special_tokens=False))
    original_token_count: int = len(tokenizer.encode(adv_prefix, add_special_tokens=False))
    
    print(f"\n===== TOKEN MINIMIZATION COMPLETE =====")
    print(f"Original token count: {original_token_count}")
    print(f"Final token count: {final_token_count}")
    print(f"Reduction: {((original_token_count - final_token_count) / original_token_count * 100):.2f}%")
    print(f"Final prefix: '{ablation_prefix}'")
    
    return ablation_prefix


def sample_control(
    control_toks: torch.Tensor, 
    grad: torch.Tensor, 
    batch_size: int, 
    topk: int = 256, 
    temp: float = 1, 
    not_allowed_tokens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = float('inf')

    top_indices: torch.Tensor = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks: torch.Tensor = control_toks.repeat(batch_size, 1)
    
    # Ensure batch_size doesn't exceed the size of control_toks
    actual_batch_size: int = min(batch_size, len(control_toks))
    
    new_token_pos: torch.Tensor = torch.arange(
        0, 
        len(control_toks), 
        max(1, len(control_toks) / actual_batch_size),  # Ensure step is at least 1
        device=grad.device
    ).type(torch.int64)
    
    # Extra safety: ensure new_token_pos is within bounds of top_indices' first dimension
    new_token_pos = torch.clamp(new_token_pos, 0, grad.shape[0] - 1)
    
    new_token_val: torch.Tensor = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (len(new_token_pos), 1), device=grad.device)
    )
    
    # Ensure we don't exceed the original batch size dimension
    new_control_toks: torch.Tensor = original_control_toks[:len(new_token_pos)].scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    return new_control_toks

def get_random_words(n: int = 10) -> List[str]:
    # pick n random words
    return random.choices(words, k=n)
    #return random.choices(bible_words, k=n)

def count_tokens(text: str, model: str = "gpt-3.5") -> int:
    """Count the number of tokens in a text string using tiktoken."""
    try:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(text))
    except:
        # Fallback to a simple approximation if tiktoken fails
        return len(text.split())

def get_combined_score(
    model: AutoModelForSequenceClassification, 
    tokenizer: AutoTokenizer, 
    text: str, 
    candidates: List[str], 
    benign_idx: int, 
    malicious_idx: int, 
    device: torch.device,
    alpha: float = 0.5, 
    token_penalty_weight: float = 0.1,
) -> int:
    """
    Evaluate multiple candidate prefixes using a combined score of loss minimization, benign maximization, and token count minimization.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    tokenizer : Tokenizer
        The tokenizer to use for text processing.
    text : str
        The target text to be classified.
    candidates : list
        List of candidate prefix strings to evaluate.
    benign_idx : int
        Index of the benign class.
    malicious_idx : int
        Index of the malicious class.
    alpha : float
        Weight for the benign score (1-alpha is weight for loss)
    token_penalty_weight : float
        Weight for token count penalty (higher values penalize longer prefixes more)

    Returns
    -------
    int
        Index of the candidate with the best combined score.
    """
    best_combined_score: float = float('-inf')
    best_idx: int = 0
    
    # Find max token count for normalization
    token_counts: List[int] = [count_tokens(candidate) for candidate in candidates]
    max_token_count: int = max(token_counts) if token_counts else 1
    
    for idx, candidate in enumerate(candidates):
        inputs: Dict[str, torch.Tensor] = tokenizer(candidate + text, return_tensors="pt")
        # Move inputs to MPS device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits: torch.Tensor = model(**inputs).logits
            probs: torch.Tensor = torch.softmax(logits, dim=-1)

            # Calculate loss component (lower is better)
            loss: torch.Tensor = nn.CrossEntropyLoss()(logits, torch.zeros(logits.shape[0], device=device).long())
            normalized_loss: float = 1.0 / (1.0 + loss.item())  # Convert to 0-1 range where higher is better

            # Calculate benign score component (higher is better)
            benign_score: float = probs[0][benign_idx].item()
            
            # Calculate token count penalty (normalized to 0-1, where higher is better = fewer tokens)
            token_count: int = token_counts[idx]
            token_penalty: float = 1.0 - (token_count / max_token_count)

            # Calculate combined score with token penalty
            combined_score: float = (alpha * benign_score + (1 - alpha) * normalized_loss) * (1 - token_penalty_weight + token_penalty_weight * token_penalty)

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_idx = idx

    return best_idx