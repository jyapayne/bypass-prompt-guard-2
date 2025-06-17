import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
import tiktoken
import random
import torch.nn as nn
from words import words4 as words
from wordsdb import WordsDatabase

# Create a global instance of the database
words_db = WordsDatabase()


def conditional_lower(text: str, lowercase_enabled: bool) -> str:
    """Apply lowercase conversion only if enabled."""
    return text.lower() if lowercase_enabled else text


def find_best_word_to_add(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    injection_text: str,
    adv_prefix: str,
    text: str,
    benign_class_idx: int,
    device: torch.device,
    num_candidates: int = 20,
    token_weight: float = 0.5,  # Weight for token count prioritization
    use_db: bool = True,  # Whether to use the database for word selection and tracking
    token_priority: float = 0.3,  # How much to prioritize words with fewer tokens when selecting from database
    order_template: str = "{injection}{prefix}{text}",  # Template for ordering components
    lowercase_enabled: bool = False,  # Whether to apply lowercase conversion
) -> Tuple[Optional[str], float]:
    """
    Evaluate multiple candidate words and find the one that most improves the benign score when added to the prefix.
    Prioritizes words that result in fewer tokens while still improving the benign score.

    Parameters:
    -----------
    model: The model to evaluate with
    tokenizer: The tokenizer to use
    injection_text: The injection text to prepend
    adv_prefix: The current prefix
    text: The text to append after the prefix
    benign_class_idx: The index of the benign class
    num_candidates: Number of candidate words to test
    token_weight: Weight for token count prioritization (higher values prioritize shorter prefixes more)
    use_db: Whether to use the database for word selection and tracking
    token_priority: How much to prioritize words with fewer tokens when selecting from database
    order_template: Template string for ordering components (using {injection}, {prefix}, {text})
    lowercase_enabled: Whether to apply lowercase conversion

    Returns:
    --------
    best_word: The word that most improves the benign score
    improvement: The amount of improvement in benign score
    """
    print(f"\n----- TESTING {num_candidates} CANDIDATE WORDS TO ADD (BATCHED) -----")

    # Get baseline benign score with current prefix
    try:
        full_text = conditional_lower(order_template.format(injection=injection_text, prefix=adv_prefix, text=text), lowercase_enabled)
        inputs: Dict[str, torch.Tensor] = tokenizer(full_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits: torch.Tensor = model(**inputs).logits
            probs: torch.Tensor = torch.softmax(logits, dim=-1)
            baseline_score: float = probs[0][benign_class_idx].item()
        print(f"Baseline benign score: {baseline_score:.4f}")
    except Exception as e:
        print(f"Error testing baseline: {e}")
        return None, 0

    # Generate candidate words to test - prioritize known good words if using database
    if use_db:
        # Try to get high-performing words from the database, with token count consideration
        db_candidates_count = num_candidates // 2
        if db_candidates_count > 0:
            top_words = words_db.get_top_words(
                limit=db_candidates_count,
                min_uses=1,  # Only need to have been tested once
                sort_by="combined" if token_priority > 0 else "improvement",
                token_weight=token_priority,
            )

            # If we got some words from the database, use them plus some random words
            if top_words:
                print(
                    f"Using {len(top_words)} words from database (with token priority {token_priority}) plus {num_candidates - len(top_words)} random words"
                )
                remaining = num_candidates - len(top_words)
                candidates = top_words + random.choices(words, k=remaining)
            else:
                # Otherwise just use random words
                candidates = random.choices(words, k=num_candidates)
        else:
            candidates = random.choices(words, k=num_candidates)
    else:
        # Just use random words if not using the database
        candidates = random.choices(words, k=num_candidates)

    # Define positions to test for each word
    insert_positions: List[str] = ["beginning", "middle", "end"]

    # Generate all candidate prefixes - one for each word + position combination
    all_candidate_prefixes = []
    for word in candidates:
        for position in insert_positions:
            # Create test prefix with the candidate word
            if position == "beginning":
                test_prefix: str = word + " " + adv_prefix
            elif position == "end":
                test_prefix = adv_prefix + " " + word
            else:  # middle
                # Find a reasonable spot to insert in the middle if possible
                if " " in adv_prefix:
                    words_list: List[str] = adv_prefix.split()
                    middle_idx: int = len(words_list) // 2
                    words_list.insert(middle_idx, word)
                    test_prefix = " ".join(words_list)
                else:
                    # If no spaces, insert at midpoint of string
                    middle_idx: int = len(adv_prefix) // 2
                    test_prefix = (
                        adv_prefix[:middle_idx] + " " + word + " " + adv_prefix[middle_idx:]
                    )

            all_candidate_prefixes.append(
                {
                    "prefix": test_prefix,
                    "word": word,
                    "position": position,
                    "token_count": len(tokenizer.encode(test_prefix, add_special_tokens=False)),
                }
            )

    # Prepare all candidate full texts for batch evaluation
    candidate_full_texts = [
        conditional_lower(order_template.format(injection=injection_text, prefix=c["prefix"], text=text), lowercase_enabled)
        for c in all_candidate_prefixes
    ]

    if not candidate_full_texts:
        print("No candidate prefixes to evaluate")
        return None, 0

    # Batch inference
    try:
        inputs = tokenizer(candidate_full_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            benign_scores = probs[:, benign_class_idx].cpu().numpy()
    except Exception as e:
        print(f"Error in batch evaluation: {e}")
        return None, 0

    # Calculate token counts for normalization
    token_counts = [c["token_count"] for c in all_candidate_prefixes]
    max_token_count = max(token_counts) if token_counts else 1

    # Process the results
    results = []
    best_combined_score = 0
    best_result_idx = -1

    print(f"Running score analysis for {len(all_candidate_prefixes)} candidate prefixes")

    for idx, candidate in enumerate(all_candidate_prefixes):
        benign_score = benign_scores[idx]
        improvement = benign_score - baseline_score
        token_count = candidate["token_count"]

        # Calculate token efficiency (lower token count is better)
        # Normalize token count to 0-1 scale (where 1 is better = fewer tokens)
        token_efficiency = 1.0 - min(1.0, token_count / max_token_count)

        # Calculate combined score (weighting improvement and token efficiency)
        # Only consider token efficiency if improvement is positive
        combined_score = 0
        if improvement > 0:
            combined_score = (1 - token_weight) * improvement + token_weight * token_efficiency

        # Record performance in results list
        result = {
            "word": candidate["word"],
            "position": candidate["position"],
            "score": benign_score,
            "improvement": improvement,
            "tokens": token_count,
            "token_efficiency": token_efficiency,
            "combined_score": combined_score,
            "prefix": candidate["prefix"],
        }

        results.append(result)

        # Record the performance in the database if enabled
        if use_db and improvement != 0:  # Only record non-zero improvements
            words_db.record_word_performance(
                candidate["word"],
                candidate["position"],
                benign_score,
                improvement,
                token_count,
                combined_score,
            )

        # print(f"Word '{candidate['word']}' at {candidate['position']}: {benign_score:.4f} (Δ: {improvement:.4f}, tokens: {token_count}, combined: {combined_score:.4f})")

        # Only consider improvements (benign_score > baseline_score)
        if improvement > 0 and combined_score > best_combined_score:
            best_combined_score = combined_score
            best_result_idx = idx

    # Sort results by combined score
    results.sort(key=lambda x: x["combined_score"], reverse=True)

    # Print top 5 results
    print("\nTop 5 most effective additions (based on combined score):")
    for i, result in enumerate(results[:5]):
        print(
            f"{i+1}. '{result['word']}' at {result['position']}: {result['score']:.4f} (Δ: {result['improvement']:.4f}, tokens: {result['tokens']}, combined: {result['combined_score']:.4f})"
        )

    if best_result_idx >= 0:
        best_result = all_candidate_prefixes[best_result_idx]
        best_word = best_result["word"]
        best_position = best_result["position"]
        best_improvement = benign_scores[best_result_idx] - baseline_score
        best_prefix = best_result["prefix"]

        print(f"\nBest word to add: '{best_word}' at {best_position}")
        print(
            f"Improvement: {best_improvement:.4f} (from {baseline_score:.4f} to {benign_scores[best_result_idx]:.4f})"
        )
        print(f"New prefix: '{best_prefix}'")
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
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=device, dtype=embed_weights.dtype),
    )
    one_hot.requires_grad_()
    input_embeds: torch.Tensor = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds: torch.Tensor = model.deberta.embeddings.word_embeddings(input_ids)
    full_embeds: torch.Tensor = torch.cat(
        [
            embeds[: input_slice.start, :],
            input_embeds.squeeze(),
            embeds[input_slice.stop :, :],
        ],
        dim=0,
    )
    logits: torch.Tensor = model(inputs_embeds=full_embeds.unsqueeze(0)).logits

    # Combined loss: minimize malicious class (standard loss) and maximize benign class
    standard_loss: torch.Tensor = nn.CrossEntropyLoss()(
        logits, torch.zeros(logits.shape[0], device=device).long()
    )

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
    device: torch.device,
    min_acceptable_benign: float = 0.6,
    order_template: str = "{injection}{prefix}{text}",  # Template for ordering components
    lowercase_enabled: bool = False,  # Whether to apply lowercase conversion
) -> str:
    """
    Simple, non-batched approach to remove as many tokens as possible while keeping
    the benign score above the minimum acceptable threshold.
    """
    print("\n----- ANALYZING TOKEN CONTRIBUTIONS (NO BATCHING) -----")

    # Get baseline benign score
    full_text = conditional_lower(order_template.format(injection=injection_text, prefix=adv_prefix, text=text), lowercase_enabled)
    inputs = tokenizer(full_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        baseline_score = probs[0][benign_class_idx].item()

    print(f"Original prefix: '{adv_prefix}'")
    print(f"Original benign score: {baseline_score:.4f}")

    # Use exactly the min_acceptable_benign as threshold
    threshold = min_acceptable_benign
    print(f"Using threshold: {threshold:.4f}")

    if baseline_score < threshold:
        print(
            f"Baseline score {baseline_score:.4f} already below threshold {threshold:.4f}. Stopping."
        )
        return adv_prefix

    current_prefix = adv_prefix
    remaining_tokens = tokenizer.tokenize(current_prefix)
    print(f"Starting with {len(remaining_tokens)} tokens")

    removed_tokens = []

    while len(remaining_tokens) > 1:
        # Try removing each token
        best_candidate = None
        best_score = -float("inf")
        best_idx = -1

        for i in range(len(remaining_tokens)):
            # Create a new candidate with this token removed
            tokens_without_i = remaining_tokens.copy()
            token_to_remove = tokens_without_i.pop(i)
            candidate_prefix = tokenizer.convert_tokens_to_string(tokens_without_i)

            # Evaluate this candidate
            full_text = conditional_lower(order_template.format(
                injection=injection_text, prefix=candidate_prefix, text=text
            ), lowercase_enabled)

            try:
                inputs = tokenizer(full_text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                    score = probs[0][benign_class_idx].item()

                print(f"  Without token {i} ('{token_to_remove}'): score = {score:.4f}")

                # If this is still above threshold and better than our current best
                if score >= threshold and score > best_score:
                    best_candidate = candidate_prefix
                    best_score = score
                    best_idx = i
                    best_token = token_to_remove
            except Exception as e:
                print(f"  Error evaluating without token {i}: {e}")

        # If we found a valid candidate, update our prefix
        if best_candidate:
            current_prefix = best_candidate
            removed_token = remaining_tokens.pop(best_idx)
            removed_tokens.append(removed_token)
            print(
                f"✓ Removed token {best_idx} ('{best_token}'): new score = {best_score:.4f}, tokens left: {len(remaining_tokens)}"
            )
        else:
            print(f"Cannot remove any more tokens while staying above threshold {threshold:.4f}")
            break

    # Final results
    print("\n===== TOKEN REMOVAL COMPLETE =====")
    print(f"Original prefix: '{adv_prefix}'")
    print(f"Final prefix: '{current_prefix}'")
    print(f"Removed {len(removed_tokens)} tokens: {removed_tokens}")
    print(f"Original token count: {len(tokenizer.tokenize(adv_prefix))}")
    print(f"Final token count: {len(remaining_tokens)}")

    # Final verification
    full_text = conditional_lower(order_template.format(injection=injection_text, prefix=current_prefix, text=text), lowercase_enabled)
    inputs = tokenizer(full_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        final_score = probs[0][benign_class_idx].item()

    print(f"Final benign score: {final_score:.4f}")

    return current_prefix


def minimize_tokens(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    injection_text: str,
    adv_prefix: str,
    text: str,
    benign_class_idx: int,
    device: torch.device,
    min_acceptable_benign: float = 0.6,
    lowercase_enabled: bool = False,  # Whether to apply lowercase conversion
) -> str:
    """
    Minimize tokens using only token contribution analysis (ablation study).
    This approach systematically removes tokens that contribute least to the benign classification.
    Prioritizes removing shorter tokens when they have similar impacts on benign score.

    Parameters:
    -----------
    token_length_weight: Weight for prioritizing removal of short tokens (0-1, higher = prioritize short tokens more)
    """
    print("\n===== STARTING TOKEN MINIMIZATION =====")

    # Use only token ablation approach - systematically remove tokens that contribute least
    ablation_prefix: str = analyze_token_contributions(
        model,
        tokenizer,
        injection_text,
        adv_prefix,
        text,
        benign_class_idx,
        device=device,
        min_acceptable_benign=min_acceptable_benign,
        lowercase_enabled=lowercase_enabled,
    )

    # Report final token count
    final_token_count: int = len(tokenizer.encode(ablation_prefix, add_special_tokens=False))
    original_token_count: int = len(tokenizer.encode(adv_prefix, add_special_tokens=False))

    print(f"\n===== TOKEN MINIMIZATION COMPLETE =====")
    print(f"Original token count: {original_token_count}")
    print(f"Final token count: {final_token_count}")
    print(
        f"Reduction: {((original_token_count - final_token_count) / original_token_count * 100):.2f}%"
    )
    print(f"Final prefix: '{ablation_prefix}'")

    return ablation_prefix


def sample_control(
    control_toks: torch.Tensor,
    grad: torch.Tensor,
    batch_size: int,
    topk: int = 256,
    temp: float = 1,
    not_allowed_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = float("inf")

    top_indices: torch.Tensor = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks: torch.Tensor = control_toks.repeat(batch_size, 1)

    # Ensure batch_size doesn't exceed the size of control_toks
    actual_batch_size: int = min(batch_size, len(control_toks))

    new_token_pos: torch.Tensor = torch.arange(
        0,
        len(control_toks),
        max(1, len(control_toks) / actual_batch_size),  # Ensure step is at least 1
        device=grad.device,
    ).type(torch.int64)

    # Extra safety: ensure new_token_pos is within bounds of top_indices' first dimension
    new_token_pos = torch.clamp(new_token_pos, 0, grad.shape[0] - 1)

    new_token_val: torch.Tensor = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (len(new_token_pos), 1), device=grad.device),
    )

    # Ensure we don't exceed the original batch size dimension
    new_control_toks: torch.Tensor = original_control_toks[: len(new_token_pos)].scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    return new_control_toks


def get_random_words(n: int = 10, min_uses: int = 0, token_priority: float = 0.3) -> List[str]:
    """
    Get a list of words to use, prioritizing words that have performed well in the past.

    Parameters:
    -----------
    n: Number of words to return
    min_uses: Minimum number of uses a word must have to be considered from the database
    token_priority: How much to prioritize words with fewer tokens (0-1)
                   0 = purely improvement based, 1 = purely token count based

    Returns:
    --------
    List of words
    """
    # Try to get high-performing words from the database
    if token_priority <= 0:
        # Sort purely by improvement
        top_words = words_db.get_top_words(limit=n, min_uses=min_uses, sort_by="improvement")
    elif token_priority >= 1:
        # Sort purely by token count (ascending)
        top_words = words_db.get_top_words(limit=n, min_uses=min_uses, sort_by="tokens")
    else:
        # Use combined sorting with the specified token weight
        top_words = words_db.get_top_words(
            limit=n, min_uses=min_uses, sort_by="combined", token_weight=token_priority
        )

    # If we got enough words from the database, use them
    if len(top_words) >= n:
        return top_words[:n]

    # Otherwise, use what we got plus some random words
    remaining = n - len(top_words)
    return top_words + random.choices(words, k=remaining)


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
    device: torch.device,
    alpha: float = 0.5,
    token_penalty_weight: float = 0.1,
    lowercase_enabled: bool = False,  # Whether to apply lowercase conversion
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
    best_combined_score: float = float("-inf")
    best_idx: int = 0

    # Find max token count for normalization
    token_counts: List[int] = [count_tokens(candidate) for candidate in candidates]
    max_token_count: int = max(token_counts) if token_counts else 1

    for idx, candidate in enumerate(candidates):
        inputs: Dict[str, torch.Tensor] = tokenizer(conditional_lower(candidate + text, lowercase_enabled), return_tensors="pt")
        # Move inputs to MPS device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits: torch.Tensor = model(**inputs).logits
            probs: torch.Tensor = torch.softmax(logits, dim=-1)

            # Calculate loss component (lower is better)
            loss: torch.Tensor = nn.CrossEntropyLoss()(
                logits, torch.zeros(logits.shape[0], device=device).long()
            )
            normalized_loss: float = 1.0 / (
                1.0 + loss.item()
            )  # Convert to 0-1 range where higher is better

            # Calculate benign score component (higher is better)
            benign_score: float = probs[0][benign_idx].item()

            # Calculate token count penalty (normalized to 0-1, where higher is better = fewer tokens)
            token_count: int = token_counts[idx]
            token_penalty: float = 1.0 - (token_count / max_token_count)

            # Calculate combined score with token penalty
            combined_score: float = (alpha * benign_score + (1 - alpha) * normalized_loss) * (
                1 - token_penalty_weight + token_penalty_weight * token_penalty
            )

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_idx = idx

    return best_idx

def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None, lowercase_enabled=False):
    def is_english_text(text):
        """Check if text contains only English characters, numbers, and spaces."""
        import re
        # Allow only English letters, numbers, and spaces,
        english_pattern = re.compile(r'^[a-zA-Z0-9\s.,?;:()\[\]{}"\'`~@#$%^&*=_\-/<>\\|]*$')
        return english_pattern.match(text) is not None
    
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = conditional_lower(tokenizer.decode(control_cand[i], skip_special_tokens=True), lowercase_enabled)
        
        # Always filter out non-English characters
        if not is_english_text(decoded_str):
            count += 1
            continue
            
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        # If we don't have enough candidates, pad with the last valid one or create a fallback
        if len(cands) == 0:
            # Fallback to current control if no valid candidates found
            fallback = curr_control if curr_control and is_english_text(curr_control) else "test"
            cands = [fallback] * len(control_cand)
        else:
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands