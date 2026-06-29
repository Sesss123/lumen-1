"""Unified 6-layer safety stack orchestrator."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from lumen.safety.alignment_verifier import AlignmentVerifier
from lumen.safety.audit_logger import AuditLogger
from lumen.safety.content_filter import ContentFilter
from lumen.safety.prompt_processor import PromptProcessor
from lumen.safety.refusal_handler import RefusalHandler


@dataclass
class SafetyResult:
    allowed: bool
    response: str
    refused: bool
    refusal_reason: Optional[str] = None
    metadata: Optional[Dict] = None


class SafetyStack:
    """
    Orchestrates all 6 safety layers:
    1. Prompt Processing
    2. Input Content Filter
    3. Model Core with Safety Head
    4. Output Content Filter
    5. Alignment Verifier
    6. Audit Logger
    """

    def __init__(
        self,
        model=None,
        verifier_model=None,
        tokenizer=None,
        audit_log_path: Optional[str] = None,
    ):
        self.model = model
        self.prompt_processor = PromptProcessor()
        self.content_filter = ContentFilter()
        self.alignment_verifier = AlignmentVerifier(verifier_model, tokenizer)
        self.refusal_handler = RefusalHandler()
        self.audit = AuditLogger(audit_log_path)

    def pre_inference(
        self,
        messages: List[Dict[str, str]],
        image_bytes: Optional[bytes] = None,
        audio_transcript: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> SafetyResult:
        # Layer 1: Prompt processing
        messages, meta = self.prompt_processor.process(messages)
        self.audit.log("prompt_process", "L1", not meta["jailbreak_detected"], meta, request_id)

        if meta["jailbreak_detected"]:
            return SafetyResult(
                allowed=False,
                response=self.refusal_handler.refuse("jailbreak"),
                refused=True,
                refusal_reason="jailbreak",
                metadata=meta,
            )

        # Layer 2: Input content filter
        combined = " ".join(m["content"] for m in messages)
        text_ok, text_meta = self.content_filter.check_text(combined)
        self.audit.log("input_text", "L2", text_ok, text_meta, request_id)
        if not text_ok:
            return SafetyResult(
                allowed=False,
                response=self.refusal_handler.refuse("harmful"),
                refused=True,
                refusal_reason="input_toxic",
                metadata=text_meta,
            )

        if image_bytes and not self.content_filter.check_image_hash(image_bytes):
            self.audit.log("input_image", "L2", False, {}, request_id)
            return SafetyResult(
                allowed=False,
                response=self.refusal_handler.refuse("harmful"),
                refused=True,
                refusal_reason="blocked_image",
            )

        if audio_transcript:
            audio_ok, audio_meta = self.content_filter.check_audio_transcript(audio_transcript)
            self.audit.log("input_audio", "L2", audio_ok, audio_meta, request_id)
            if not audio_ok:
                return SafetyResult(
                    allowed=False,
                    response=self.refusal_handler.refuse("harmful"),
                    refused=True,
                    refusal_reason="input_audio_toxic",
                )

        return SafetyResult(allowed=True, response="", refused=False, metadata=meta)

    def post_inference(
        self,
        prompt: str,
        response: str,
        safety_logits: Optional[torch.Tensor] = None,
        request_id: Optional[str] = None,
    ) -> SafetyResult:
        # Layer 3: In-model safety head
        if safety_logits is not None:
            reason = self.refusal_handler.should_refuse_safety_head(safety_logits)
            self.audit.log("safety_head", "L3", reason is None, {"reason": reason}, request_id)
            if reason:
                return SafetyResult(
                    allowed=False,
                    response=self.refusal_handler.refuse(reason),
                    refused=True,
                    refusal_reason=f"safety_head_{reason}",
                )

        # Layer 4: Output content filter
        filtered, out_ok = self.content_filter.filter_output(response)
        self.audit.log("output_filter", "L4", out_ok, {}, request_id)
        if not out_ok:
            return SafetyResult(
                allowed=False,
                response=self.refusal_handler.refuse("harmful"),
                refused=True,
                refusal_reason="output_blocked",
            )

        # Layer 5: Alignment verifier
        aligned, align_meta = self.alignment_verifier.verify(prompt, filtered)
        self.audit.log("alignment", "L5", aligned, align_meta, request_id)
        if not aligned:
            return SafetyResult(
                allowed=False,
                response=self.refusal_handler.refuse("default"),
                refused=True,
                refusal_reason="alignment_failed",
                metadata=align_meta,
            )

        # Layer 6: Audit (logged throughout)
        return SafetyResult(allowed=True, response=filtered, refused=False, metadata=align_meta)

    def wrap_generate(self, engine, messages: List[Dict], **kwargs) -> SafetyResult:
        """Full pipeline: pre-check -> generate -> post-check."""
        pre = self.pre_inference(messages)
        if not pre.allowed:
            return pre

        prompt = " ".join(m["content"] for m in messages)
        response = engine.generate(messages, **kwargs)

        safety_logits = None
        if self.model is not None:
            with torch.no_grad():
                ids = engine.tokenizer.encode(prompt + response)
                out = self.model(
                    torch.tensor([ids], device=engine.device),
                    return_safety_logits=True,
                )
                safety_logits = out.safety_logits

        return self.post_inference(prompt, response, safety_logits)
