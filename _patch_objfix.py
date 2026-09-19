# -*- coding: utf-8 -*-
# Reviving object detection exposed latent bugs in the object_detected path
# (never exercised before because detection never produced events):
#  - it emitted ~1 event/frame -> flood
#  - main._handle_object_detected called save_object(confidence=...) (unsupported)
#    on a wrong data shape
#  - robot_brain._handle_object_detection read obj['class'] (event sends 'label')
# Fix: throttle emission (scene-change or every 5s) + fix both handlers.

# 1) throttle emission
o="modules/vision/object_detection.py"
s=open(o,encoding="utf-8").read(); orig=s
old_emit='''        # Aggregate by label for a cleaner event
        summary: Dict[str, int] = defaultdict(int)
        for det in detections:
            summary[det.label] += 1

        event = RobotEvent(
            type="object_detected",
            source="object_detection",
            data={
                "objects": [
                    {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}
                    for d in detections
                ],
                "summary": dict(summary),
            },
            priority=6,
        )'''
new_emit='''        # Aggregate by label for a cleaner event
        summary: Dict[str, int] = defaultdict(int)
        for det in detections:
            summary[det.label] += 1
        summary = dict(summary)

        # Throttle: only emit when the scene (labels+counts) changes, or every
        # ~5s. Otherwise we flood the brain with ~1 event per processed frame.
        now = time.time()
        if summary == getattr(self, "_last_emit_summary", None) and \\
                (now - getattr(self, "_last_emit_time", 0.0)) < 5.0:
            return
        self._last_emit_summary = summary
        self._last_emit_time = now

        event = RobotEvent(
            type="object_detected",
            source="object_detection",
            data={
                "objects": [
                    {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}
                    for d in detections
                ],
                "summary": summary,
            },
            priority=6,
        )'''
assert old_emit in s, "emit anchor not found"
s=s.replace(old_emit,new_emit,1)
open(o,"w",encoding="utf-8").write(s)
print("object_detection throttle:", s!=orig)

# 2) fix main handler
m="main.py"
s=open(m,encoding="utf-8").read(); orig=s
old_main='''        obj_data = event.data
        label = obj_data.get('label', 'unknown')
        confidence = obj_data.get('confidence', 0)
        self.logger.debug(f"Object detected: {label} ({confidence:.0%})")

        if self.learning_db:
            self.learning_db.save_object(label, confidence=confidence)'''
new_main='''        data = event.data or {}
        summary = data.get('summary') or {}
        objs = data.get('objects') or []
        if objs:
            self.logger.debug("Objects seen: %s",
                              summary or [o.get('label') for o in objs])
        # No per-event DB writes: object detection runs continuously, so saving
        # every scene would spam the learning DB. Deliberate object learning
        # happens via the explicit learning path in the brain.'''
assert old_main in s, "main handler anchor not found"
s=s.replace(old_main,new_main,1)
open(m,"w",encoding="utf-8").write(s)
print("main handler fixed:", s!=orig)

# 3) fix brain handler (core/robot_brain.py)
b="core/robot_brain.py"
s=open(b,encoding="utf-8").read(); orig=s
old_brain='''        for obj in objects:
            # Add to working memory
            self.working_memory[f"object_{obj['class']}"] = {
                'location': obj.get('location'),
                'confidence': obj.get('confidence'),
                'time': time.time()
            }

            # Check if this is what we're looking for
            if self.current_task and self.current_task.get('type') == 'find_object':
                target = self.current_task.get('target')
                if target and target.lower() in obj['class'].lower():'''
new_brain='''        for obj in objects:
            label = obj.get('label') or obj.get('class') or 'object'
            # Add to working memory
            self.working_memory[f"object_{label}"] = {
                'location': obj.get('location') or obj.get('bbox'),
                'confidence': obj.get('confidence'),
                'time': time.time()
            }

            # Check if this is what we're looking for
            if self.current_task and self.current_task.get('type') == 'find_object':
                target = self.current_task.get('target')
                if target and target.lower() in label.lower():'''
assert old_brain in s, "brain handler anchor not found"
s=s.replace(old_brain,new_brain,1)
open(b,"w",encoding="utf-8").write(s)
print("brain handler fixed:", s!=orig)
