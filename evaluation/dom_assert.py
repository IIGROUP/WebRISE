from __future__ import annotations

from typing import Any


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


async def start_dom_monitor(page, target_selector: str | None = None, observed_selectors: dict | None = None) -> None:
    await page.evaluate(
        """
        ({targetSelector, observedSelectors}) => {
          const normalize = (text) => (text || '').replace(/\\s+/g, ' ').trim();
          const MAX_FIELD_CHARS = 4000;
          const sanitizeValue = (value) => {
            if (value == null) return value;
            const str = String(value);
            if (str.startsWith('data:')) {
              const comma = str.indexOf(',');
              const head = comma >= 0 ? str.slice(0, Math.min(comma, 120)) : str.slice(0, 120);
              return `${head},...[data-url ${str.length} chars]`;
            }
            if (str.length > MAX_FIELD_CHARS) {
              return `${str.slice(0, MAX_FIELD_CHARS)}...[truncated ${str.length} chars]`;
            }
            return str;
          };
          const normalizeSafe = (text) => {
            const value = sanitizeValue(text);
            return value == null ? '' : normalize(value);
          };
          const BOOLEAN_ATTRS = new Set(['disabled', 'hidden', 'checked', 'selected', 'open', 'multiple', 'readonly', 'required']);
          const ARIA_BOOLEAN_ATTRS = new Set(['aria-disabled', 'aria-hidden', 'aria-busy', 'aria-expanded', 'aria-checked', 'aria-selected']);

          const isVisible = (el) => {
            if (!el) return false;
            for (let node = el; node && node.nodeType === Node.ELEMENT_NODE; node = node.parentElement) {
              const style = window.getComputedStyle(node);
              if (
                style.display === 'none' ||
                style.visibility === 'hidden' ||
                style.opacity === '0' ||
                node.hasAttribute('hidden')
              ) {
                return false;
              }
            }
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
          };

          const collectPageTextWithVisibility = () => {
            if (!document.body) return '';
            const parts = [];
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            while (walker.nextNode()) {
              const text = normalize(walker.currentNode.textContent || '');
              if (!text) continue;
              const parent = walker.currentNode.parentElement;
              if (!parent || ['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE'].includes(parent.tagName)) continue;
              parts.push((isVisible(parent) ? '' : '[not-visible] ') + text);
            }
            for (const el of document.querySelectorAll('select, input, textarea')) {
              if (!isVisible(el)) continue;
              const label = normalize(el.getAttribute('aria-label') || el.getAttribute('name') || '');
              let value = '';
              if (el.tagName === 'SELECT' && el.selectedIndex >= 0) {
                value = normalize(el.options[el.selectedIndex].text);
              } else {
                value = normalize(el.value || '');
              }
              if (value) parts.push(label ? `${label}: ${value}` : value);
            }
            return normalize(parts.join(' '));
          };

          const getTargetState = (selector) => {
            if (!selector) return null;
            const el = document.querySelector(selector);
            if (!el) {
              return {
                exists: false,
                visible: false,
                disabled: false,
                text: ''
              };
            }
            const ariaDisabled = el.getAttribute('aria-disabled') === 'true';
            const disabled = typeof el.disabled !== 'undefined' ? !!el.disabled : ariaDisabled;
            const getAriaBool = (attr) => {
              const v = el.getAttribute(attr);
              return v === null ? null : v === 'true';
            };
            const selectedText = el.tagName === 'SELECT' && el.selectedIndex >= 0
              ? normalize(el.options[el.selectedIndex].text)
              : null;
            return {
              exists: true,
              visible: isVisible(el),
              disabled,
              pointer_events: window.getComputedStyle(el).pointerEvents,
              text: normalizeSafe(el.value !== undefined && el.value !== '' ? el.value : (el.innerText || el.textContent || '')),
              class: normalizeSafe(el.className || ''),
              value: el.value !== undefined ? sanitizeValue(el.value) : null,
              min: el.getAttribute('min') || '',
              max: el.getAttribute('max') || '',
              selected_text: selectedText,
              aria_label: el.getAttribute('aria-label') || '',
              aria_selected: getAriaBool('aria-selected'),
              aria_expanded: getAriaBool('aria-expanded'),
              aria_pressed: getAriaBool('aria-pressed'),
              aria_checked: typeof el.checked !== 'undefined' ? !!el.checked : getAriaBool('aria-checked'),
            };
          };

          const normalizeAttrValue = (value) => value == null ? null : normalize(sanitizeValue(value));

          const summarizeAttributeMutation = (m) => {
            const el = m.target && m.target.nodeType === Node.ELEMENT_NODE ? m.target : null;
            if (!el) return null;
            const attr = m.attributeName || '';
            const oldValue = normalizeAttrValue(m.oldValue);
            const newValue = normalizeAttrValue(el.getAttribute(attr));
            if (oldValue === newValue) return null;
            const summary = {
              attribute: attr,
              old_value: oldValue,
              new_value: newValue,
              node: summarizeNode(el)
            };
            if (BOOLEAN_ATTRS.has(attr)) {
              summary.old_present = m.oldValue !== null;
              summary.new_present = el.hasAttribute(attr);
            } else if (ARIA_BOOLEAN_ATTRS.has(attr)) {
              summary.old_bool = String(oldValue || '').toLowerCase() === 'true';
              summary.new_bool = String(newValue || '').toLowerCase() === 'true';
            }
            return summary;
          };

          const snapshot = () => {
            const snap = {
              page_text: collectPageTextWithVisibility()
            };
            if (targetSelector) snap.target = getTargetState(targetSelector);
            return snap;
          };

          // Full interactive-element scan — only called for initial and final
          // snapshots (not per-event) to avoid bloating the event log.
          const snapshotWithElements = () => {
            const snap = snapshot();
            try {
              const interactiveSelectors = 'button, input, select, textarea, a[href], [onclick], [contenteditable="true"], [tabindex]:not([tabindex="-1"]), [role="button"], [role="tab"], [role="checkbox"], [role="switch"], [role="radio"], [role="option"], [role="menuitem"], [role="slider"]';
              const trackedEvents = new Set(['click', 'mousedown', 'mouseup', 'dblclick', 'contextmenu', 'pointerdown', 'pointerup']);
              const baseElements = Array.from(document.querySelectorAll(interactiveSelectors));
              const trackedElements = Array.from(document.querySelectorAll('*')).filter((el) => (
                el.__trackedEvents && Array.from(el.__trackedEvents).some((type) => trackedEvents.has(type))
              ));
              const elements = Array.from(new Set([...baseElements, ...trackedElements]));
              const states = [];
              for (let i = 0; i < elements.length && states.length < 100; i++) {
                const el = elements[i];
                if (!el || !isVisible(el)) continue;
                const getAriaBool = (attr) => {
                  const v = el.getAttribute(attr);
                  return v === null ? null : v === 'true';
                };
                const ariaDisabled = el.getAttribute('aria-disabled') === 'true';
                const selectedText = el.tagName === 'SELECT' && el.selectedIndex >= 0
                  ? normalize(el.options[el.selectedIndex].text) : null;
                states.push({
                  tag: (el.tagName || '').toLowerCase(),
                  id: el.id || '',
                  class: normalizeSafe(el.className || ''),
                  text: normalizeSafe(el.value !== undefined && el.value !== '' ? el.value : (el.innerText || el.textContent || '')),
                  visible: true,
                  disabled: typeof el.disabled !== 'undefined' ? !!el.disabled : ariaDisabled,
                  pointer_events: window.getComputedStyle(el).pointerEvents,
                  value: el.value !== undefined ? sanitizeValue(el.value) : null,
                  min: el.getAttribute('min') || '',
                  max: el.getAttribute('max') || '',
                  selected_text: selectedText,
                  aria_label: el.getAttribute('aria-label') || '',
                  aria_selected: getAriaBool('aria-selected'),
                  aria_expanded: getAriaBool('aria-expanded'),
                  aria_pressed: getAriaBool('aria-pressed'),
                  aria_checked: typeof el.checked !== 'undefined' ? !!el.checked : getAriaBool('aria-checked'),
                });
              }
              snap.interactive_elements = states;
            } catch (_) {}
            return snap;
          };

          const collectDescendantSignatures = (el) => {
            // Returns a map of "tag.class" → count for all descendants, so the
            // scorer can see inner structure (e.g. skeleton-card) that would
            // otherwise be hidden when the mutation added only an outer wrapper.
            const counts = {};
            try {
              const all = el.querySelectorAll('*');
              for (let i = 0; i < all.length; i++) {
                const d = all[i];
                const tag = (d.tagName || '').toLowerCase();
                if (!tag) continue;
                const cls = normalize(d.className || '').split(/\\s+/).filter(Boolean);
                const sig = cls.length ? (tag + '.' + cls.join('.')) : tag;
                counts[sig] = (counts[sig] || 0) + 1;
              }
            } catch (_) {}
            return counts;
          };

          const summarizeNode = (node) => {
            try {
            if (!node) return null;
            if (node.nodeType === Node.TEXT_NODE) {
              const text = normalize(node.textContent || '');
              if (!text) return null;
              const parent = node.parentElement;
              return parent ? summarizeNode(parent) : null;
            }
            if (node.nodeType !== Node.ELEMENT_NODE) return null;
            const el = node;
            const text = normalizeSafe(el.innerText || el.textContent || '');
            const isAnchorWithDownload = el.tagName === 'A' && el.hasAttribute('download');
            const descendants = collectDescendantSignatures(el);
            const getAriaBool = (attr) => {
              const v = el.getAttribute(attr);
              return v === null ? null : v === 'true';
            };
            const ariaDisabled = el.getAttribute('aria-disabled') === 'true';
            const disabled = typeof el.disabled !== 'undefined' ? !!el.disabled : ariaDisabled;
            let pointerEvents = null;
            try { pointerEvents = window.getComputedStyle(el).pointerEvents; } catch (_) {}
            return {
              tag: (el.tagName || '').toLowerCase(),
              id: el.id || '',
              role: el.getAttribute('role') || '',
              class: normalizeSafe(el.className || ''),
              aria_label: el.getAttribute('aria-label') || '',
              data_attrs: Object.fromEntries(
                Array.from(el.attributes || [])
                  .filter((attr) => attr && attr.name && attr.name.startsWith('data-'))
                  .slice(0, 10)
                  .map((attr) => [attr.name, sanitizeValue(attr.value || '')])
              ),
              text,
              visible: isVisible(el),
              disabled,
              pointer_events: pointerEvents,
              value: el.value !== undefined ? sanitizeValue(el.value) : null,
              min: el.getAttribute('min') || '',
              max: el.getAttribute('max') || '',
              aria_selected: getAriaBool('aria-selected'),
              aria_expanded: getAriaBool('aria-expanded'),
              aria_pressed: getAriaBool('aria-pressed'),
              aria_checked: typeof el.checked !== 'undefined' ? !!el.checked : getAriaBool('aria-checked'),
              descendant_signatures: Object.keys(descendants).length ? descendants : undefined,
              download_attr: isAnchorWithDownload ? (el.getAttribute('download') || '') : undefined,
              href: isAnchorWithDownload ? ((el.href || '').startsWith('blob:') ? 'blob' : (el.href || '').startsWith('data:') ? 'data' : (el.href || '').substring(0, 80)) : undefined
            };
            } catch (e) {
              if (!window.__summarizeNodeErrors) window.__summarizeNodeErrors = [];
              window.__summarizeNodeErrors.push({msg: e.message, tag: (node.tagName||''), class: (node.className||''), stack: (e.stack||'').substring(0,150)});
              return null;
            }
          };

          const summarizeTargetTransition = (beforeSnap, afterSnap) => {
            const before = beforeSnap && beforeSnap.target ? beforeSnap.target : null;
            const after = afterSnap && afterSnap.target ? afterSnap.target : null;
            if (!before && !after) return null;
            return {
              before: before || null,
              after: after || null,
              changed: {
                exists: !!before && !!after ? before.exists !== after.exists : before !== after,
                visible: !!before && !!after ? before.visible !== after.visible : before !== after,
                disabled: !!before && !!after ? before.disabled !== after.disabled : before !== after,
                text: !!before && !!after ? normalize(before.text) !== normalize(after.text) : before !== after,
                class: !!before && !!after ? normalize(before.class) !== normalize(after.class) : before !== after,
                value: !!before && !!after ? before.value !== after.value : before !== after,
                aria_selected: !!before && !!after ? before.aria_selected !== after.aria_selected : before !== after,
                aria_expanded: !!before && !!after ? before.aria_expanded !== after.aria_expanded : before !== after,
                aria_pressed: !!before && !!after ? before.aria_pressed !== after.aria_pressed : before !== after,
                aria_checked: !!before && !!after ? before.aria_checked !== after.aria_checked : before !== after
              }
            };
          };

          const summarizeRawMutation = (m) => {
            if (!m) return null;
            const targetNode = summarizeNode(m.target && m.target.nodeType === Node.TEXT_NODE ? m.target.parentElement : m.target);
            if (m.type === 'childList') {
              return {
                type: 'childList',
                target: targetNode,
                added_nodes: Array.from(m.addedNodes || []).map(summarizeNode).filter(Boolean),
                removed_nodes: Array.from(m.removedNodes || []).map(summarizeNode).filter(Boolean),
                previous_sibling: summarizeNode(m.previousSibling),
                next_sibling: summarizeNode(m.nextSibling)
              };
            }
            if (m.type === 'attributes') {
              const attrSummary = summarizeAttributeMutation(m);
              return {
                type: 'attributes',
                target: targetNode,
                attribute_name: m.attributeName || '',
                old_value: attrSummary ? attrSummary.old_value : null,
                new_value: attrSummary ? attrSummary.new_value : null,
                old_present: attrSummary && Object.prototype.hasOwnProperty.call(attrSummary, 'old_present') ? attrSummary.old_present : null,
                new_present: attrSummary && Object.prototype.hasOwnProperty.call(attrSummary, 'new_present') ? attrSummary.new_present : null,
                old_bool: attrSummary && Object.prototype.hasOwnProperty.call(attrSummary, 'old_bool') ? attrSummary.old_bool : null,
                new_bool: attrSummary && Object.prototype.hasOwnProperty.call(attrSummary, 'new_bool') ? attrSummary.new_bool : null
              };
            }
            if (m.type === 'characterData') {
              return {
                type: 'characterData',
                target: targetNode,
                old_value: normalizeAttrValue(m.oldValue),
                new_value: normalizeAttrValue(m.target && m.target.textContent ? m.target.textContent : '')
              };
            }
            return {
              type: m.type || 'unknown',
              target: targetNode
            };
          };

          if (window.__webevalDomMonitor?.observer) {
            try { window.__webevalDomMonitor.observer.disconnect(); } catch (_) {}
            if (window.__webevalDomMonitor._downloadClickHandler) {
              try { document.removeEventListener('click', window.__webevalDomMonitor._downloadClickHandler, true); } catch (_) {}
            }
          }

          const initialObserved = Object.fromEntries(
            Object.entries(observedSelectors || {}).map(([id, sel]) => [id, getTargetState(sel)])
          );
          const state = {
            startedAt: Date.now(),
            targetSelector,
            observedSelectors: observedSelectors || {},
            initial_observed: initialObserved,
            events: [],
            initial_snapshot: snapshotWithElements(),
            final_snapshot: null,
            last_snapshot: null,
            observer: null
          };
          state.last_snapshot = state.initial_snapshot;

          const record = (types, mutations) => {
            try {
            const beforeSnapshot = state.last_snapshot || state.initial_snapshot;
            const currentSnapshot = snapshot();
            const addedNodes = [];
            const removedNodes = [];
            const changedAttributes = [];
            for (const m of mutations) {
              if (m.type === 'childList') {
                for (const node of Array.from(m.addedNodes || [])) {
                  const summary = summarizeNode(node);
                  if (summary) addedNodes.push(summary);
                }
                for (const node of Array.from(m.removedNodes || [])) {
                  const summary = summarizeNode(node);
                  if (summary) removedNodes.push(summary);
                }
              } else if (m.type === 'attributes') {
                const summary = summarizeAttributeMutation(m);
                if (summary) changedAttributes.push(summary);
              } else if (m.type === 'characterData') {
                const parent = m.target && m.target.parentElement ? m.target.parentElement : null;
                const summary = parent ? summarizeNode(parent) : null;
                changedAttributes.push({
                  attribute: 'characterData',
                  old_value: normalizeAttrValue(m.oldValue),
                  new_value: normalizeAttrValue(m.target && m.target.textContent ? m.target.textContent : ''),
                  node: summary
                });
              }
            }
            const compressNodes = (nodes) => {
              if (nodes.length <= 30) return nodes;
              const groups = {};
              for (const n of nodes) {
                const key = (n.tag || '') + '.' + (n.class || '').split(/\\s+/).sort().join('.');
                if (!groups[key]) groups[key] = { sample: n, count: 0 };
                groups[key].count++;
              }
              const result = [];
              for (const key of Object.keys(groups)) {
                const g = groups[key];
                if (g.count <= 3) {
                  for (const n of nodes) {
                    const nk = (n.tag || '') + '.' + (n.class || '').split(/\\s+/).sort().join('.');
                    if (nk === key) result.push(n);
                  }
                } else {
                  const s = Object.assign({}, g.sample);
                  s.text = (s.text || '').substring(0, 40);
                  s._compressed_count = g.count;
                  result.push(s);
                }
              }
              return result;
            };
            const evt = {
              t_ms: Date.now() - state.startedAt,
              kind: 'mutation_batch',
              mutation_types: Array.from(new Set(types)),
              added_nodes: compressNodes(addedNodes),
              removed_nodes: compressNodes(removedNodes),
              changed_attributes: changedAttributes
            };
            if (targetSelector) {
              evt.target_transition = summarizeTargetTransition(beforeSnapshot, currentSnapshot);
            }
            state.events.push(evt);
            state.last_snapshot = currentSnapshot;
            } catch (e) { state.events.push({t_ms: Date.now() - state.startedAt, kind: 'record_error', error: e.message}); }
          };

          const observer = new MutationObserver((mutations) => {
            const types = [];
            for (const m of mutations) {
              if (m.type === 'attributes') {
                types.push(`attributes:${m.attributeName || ''}`);
              } else {
                types.push(m.type);
              }
            }
            record(types, mutations);
          });

          observer.observe(document.documentElement, {
            subtree: true,
            childList: true,
            attributes: true,
            characterData: true,
            attributeOldValue: true,
            characterDataOldValue: true
          });

          state.observer = observer;

          // Intercept programmatic clicks on download anchors (capture phase catches .click() calls too)
          const _downloadClickHandler = (e) => {
            const a = e.target && e.target.closest ? e.target.closest('a[download]') : null;
            if (a) {
              state.events.push({
                t_ms: Date.now() - state.startedAt,
                kind: 'download_triggered',
                filename: a.getAttribute('download') || '',
                href_type: (a.href || '').startsWith('blob:') ? 'blob' : (a.href || '').startsWith('data:') ? 'data' : 'url',
              });
            }
          };
          document.addEventListener('click', _downloadClickHandler, true);
          state._downloadClickHandler = _downloadClickHandler;

          window.__webevalDomMonitor = state;
        }
        """,
        {"targetSelector": target_selector, "observedSelectors": observed_selectors or {}},
    )

async def finish_dom_monitor(page, wait_ms: int = 0) -> dict[str, Any]:
    if wait_ms and wait_ms > 0:
        await page.wait_for_timeout(wait_ms)

    result = await page.evaluate(
        """
        () => {
          const normalize = (text) => (text || '').replace(/\\s+/g, ' ').trim();
          const MAX_FIELD_CHARS = 4000;
          const sanitizeValue = (value) => {
            if (value == null) return value;
            const str = String(value);
            if (str.startsWith('data:')) {
              const comma = str.indexOf(',');
              const head = comma >= 0 ? str.slice(0, Math.min(comma, 120)) : str.slice(0, 120);
              return `${head},...[data-url ${str.length} chars]`;
            }
            if (str.length > MAX_FIELD_CHARS) {
              return `${str.slice(0, MAX_FIELD_CHARS)}...[truncated ${str.length} chars]`;
            }
            return str;
          };
          const normalizeSafe = (text) => {
            const value = sanitizeValue(text);
            return value == null ? '' : normalize(value);
          };

          const isVisible = (el) => {
            if (!el) return false;
            for (let node = el; node && node.nodeType === Node.ELEMENT_NODE; node = node.parentElement) {
              const style = window.getComputedStyle(node);
              if (
                style.display === 'none' ||
                style.visibility === 'hidden' ||
                style.opacity === '0' ||
                node.hasAttribute('hidden')
              ) {
                return false;
              }
            }
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
          };

          const collectPageTextWithVisibility = () => {
            if (!document.body) return '';
            const parts = [];
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            while (walker.nextNode()) {
              const text = normalize(walker.currentNode.textContent || '');
              if (!text) continue;
              const parent = walker.currentNode.parentElement;
              if (!parent || ['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE'].includes(parent.tagName)) continue;
              parts.push((isVisible(parent) ? '' : '[not-visible] ') + text);
            }
            for (const el of document.querySelectorAll('select, input, textarea')) {
              if (!isVisible(el)) continue;
              const label = normalize(el.getAttribute('aria-label') || el.getAttribute('name') || '');
              let value = '';
              if (el.tagName === 'SELECT' && el.selectedIndex >= 0) {
                value = normalize(el.options[el.selectedIndex].text);
              } else {
                value = normalize(el.value || '');
              }
              if (value) parts.push(label ? `${label}: ${value}` : value);
            }
            return normalize(parts.join(' '));
          };

          const state = window.__webevalDomMonitor;
          if (!state) {
            return {
              target_selector: null,
              initial_snapshot: { page_text: '' },
              events: [],
              final_snapshot: { page_text: '' }
            };
          }

          const getTargetState = (selector) => {
            if (!selector) return null;
            const el = document.querySelector(selector);
            if (!el) {
              return {
                exists: false,
                visible: false,
                disabled: false,
                text: ''
              };
            }
            const ariaDisabled = el.getAttribute('aria-disabled') === 'true';
            const disabled = typeof el.disabled !== 'undefined' ? !!el.disabled : ariaDisabled;
            const getAriaBool = (attr) => {
              const v = el.getAttribute(attr);
              return v === null ? null : v === 'true';
            };
            const selectedText = el.tagName === 'SELECT' && el.selectedIndex >= 0
              ? normalize(el.options[el.selectedIndex].text)
              : null;
            return {
              exists: true,
              visible: isVisible(el),
              disabled,
              pointer_events: window.getComputedStyle(el).pointerEvents,
              text: normalizeSafe(el.value !== undefined && el.value !== '' ? el.value : (el.innerText || el.textContent || '')),
              class: normalizeSafe(el.className || ''),
              value: el.value !== undefined ? sanitizeValue(el.value) : null,
              min: el.getAttribute('min') || '',
              max: el.getAttribute('max') || '',
              selected_text: selectedText,
              aria_label: el.getAttribute('aria-label') || '',
              aria_selected: getAriaBool('aria-selected'),
              aria_expanded: getAriaBool('aria-expanded'),
              aria_pressed: getAriaBool('aria-pressed'),
              aria_checked: typeof el.checked !== 'undefined' ? !!el.checked : getAriaBool('aria-checked'),
            };
          };

          const finalSnapshot = {
            page_text: collectPageTextWithVisibility()
          };
          if (state.targetSelector) {
            finalSnapshot.target = getTargetState(state.targetSelector);
          }
          // Interactive element snapshot (mirrors start_dom_monitor's snapshot())
          try {
            const interactiveSelectors = 'button, input, select, textarea, a[href], [onclick], [contenteditable="true"], [tabindex]:not([tabindex="-1"]), [role="button"], [role="tab"], [role="checkbox"], [role="switch"], [role="radio"], [role="option"], [role="menuitem"], [role="slider"]';
            const trackedEvents = new Set(['click', 'mousedown', 'mouseup', 'dblclick', 'contextmenu', 'pointerdown', 'pointerup']);
            const baseElements = Array.from(document.querySelectorAll(interactiveSelectors));
            const trackedElements = Array.from(document.querySelectorAll('*')).filter((el) => (
              el.__trackedEvents && Array.from(el.__trackedEvents).some((type) => trackedEvents.has(type))
            ));
            const elements = Array.from(new Set([...baseElements, ...trackedElements]));
            const states = [];
            for (let i = 0; i < elements.length && states.length < 100; i++) {
              const el = elements[i];
              if (!el || !isVisible(el)) continue;
              const getAriaBool2 = (attr) => {
                const v = el.getAttribute(attr);
                return v === null ? null : v === 'true';
              };
              const ariaDisabled2 = el.getAttribute('aria-disabled') === 'true';
              const selectedText2 = el.tagName === 'SELECT' && el.selectedIndex >= 0
                ? normalize(el.options[el.selectedIndex].text) : null;
              states.push({
                tag: (el.tagName || '').toLowerCase(),
                id: el.id || '',
                class: normalizeSafe(el.className || ''),
                text: normalizeSafe(el.value !== undefined && el.value !== '' ? el.value : (el.innerText || el.textContent || '')),
                visible: true,
                disabled: typeof el.disabled !== 'undefined' ? !!el.disabled : ariaDisabled2,
                pointer_events: window.getComputedStyle(el).pointerEvents,
                value: el.value !== undefined ? sanitizeValue(el.value) : null,
                min: el.getAttribute('min') || '',
                max: el.getAttribute('max') || '',
                selected_text: selectedText2,
                aria_label: el.getAttribute('aria-label') || '',
                aria_selected: getAriaBool2('aria-selected'),
                aria_expanded: getAriaBool2('aria-expanded'),
                aria_pressed: getAriaBool2('aria-pressed'),
                aria_checked: typeof el.checked !== 'undefined' ? !!el.checked : getAriaBool2('aria-checked'),
              });
            }
            finalSnapshot.interactive_elements = states;
          } catch (_) {}

          try { state.observer?.disconnect(); } catch (_) {}
          if (state._downloadClickHandler) {
            try { document.removeEventListener('click', state._downloadClickHandler, true); } catch (_) {}
          }

          window.__webevalDomMonitor = null;

          const result = {
            target_selector: state.targetSelector,
            initial_snapshot: state.initial_snapshot,
            events: state.events,
            final_snapshot: finalSnapshot
          };
          const observedSelectors = state.observedSelectors || {};
          if (Object.keys(observedSelectors).length > 0) {
            const observedElements = {};
            for (const [id, sel] of Object.entries(observedSelectors)) {
              const before = (state.initial_observed || {})[id] || null;
              const after = getTargetState(sel);
              const allKeys = (before || after)
                ? [...new Set([...Object.keys(before || {}), ...Object.keys(after || {})])]
                : [];
              const changed = (before && after)
                ? Object.fromEntries(allKeys.map(k => [k, before[k] !== after[k]]))
                : null;
              observedElements[id] = { before, after, changed };
            }
            result.observed_elements = observedElements;
          }
          return result;
        }
        """
    )

    return result


async def capture_snapshot(page, target_selector: str | None, observed_selectors: dict | None = None) -> dict[str, Any]:
    """Take a one-shot target + observed snapshot without touching window state.

    Used as a before/after substitute for actions that destroy the DOM monitor
    (e.g. `refresh`, full navigations). The returned dict has the same shape as
    the target/observed entries produced inside `finish_dom_monitor`, so the
    scorer's field-meaning prompts continue to apply.
    """
    return await page.evaluate(
        """
        (params) => {
          const { targetSelector, observedSelectors } = params || {};
          const normalize = (text) => (text || '').replace(/\\s+/g, ' ').trim();
          const MAX_FIELD_CHARS = 4000;
          const sanitizeValue = (value) => {
            if (value == null) return value;
            const str = String(value);
            if (str.startsWith('data:')) {
              const comma = str.indexOf(',');
              const head = comma >= 0 ? str.slice(0, Math.min(comma, 120)) : str.slice(0, 120);
              return `${head},...[data-url ${str.length} chars]`;
            }
            if (str.length > MAX_FIELD_CHARS) {
              return `${str.slice(0, MAX_FIELD_CHARS)}...[truncated ${str.length} chars]`;
            }
            return str;
          };
          const normalizeSafe = (text) => {
            const value = sanitizeValue(text);
            return value == null ? '' : normalize(value);
          };
          const isVisible = (el) => {
            if (!el) return false;
            for (let node = el; node && node.nodeType === Node.ELEMENT_NODE; node = node.parentElement) {
              const style = window.getComputedStyle(node);
              if (
                style.display === 'none' ||
                style.visibility === 'hidden' ||
                style.opacity === '0' ||
                node.hasAttribute('hidden')
              ) return false;
            }
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
          };
          const collectPageTextWithVisibility = () => {
            if (!document.body) return '';
            const parts = [];
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            while (walker.nextNode()) {
              const text = normalize(walker.currentNode.textContent || '');
              if (!text) continue;
              const parent = walker.currentNode.parentElement;
              if (!parent || ['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE'].includes(parent.tagName)) continue;
              parts.push((isVisible(parent) ? '' : '[not-visible] ') + text);
            }
            for (const el of document.querySelectorAll('select, input, textarea')) {
              if (!isVisible(el)) continue;
              const label = normalize(el.getAttribute('aria-label') || el.getAttribute('name') || '');
              let value = '';
              if (el.tagName === 'SELECT' && el.selectedIndex >= 0) {
                value = normalize(el.options[el.selectedIndex].text);
              } else {
                value = normalize(el.value || '');
              }
              if (value) parts.push(label ? `${label}: ${value}` : value);
            }
            return normalize(parts.join(' '));
          };
          const getTargetState = (selector) => {
            if (!selector) return null;
            const el = document.querySelector(selector);
            if (!el) return { exists: false, visible: false, disabled: false, text: '' };
            const ariaDisabled = el.getAttribute('aria-disabled') === 'true';
            const disabled = typeof el.disabled !== 'undefined' ? !!el.disabled : ariaDisabled;
            const getAriaBool = (attr) => {
              const v = el.getAttribute(attr);
              return v === null ? null : v === 'true';
            };
            const selectedText = el.tagName === 'SELECT' && el.selectedIndex >= 0
              ? normalize(el.options[el.selectedIndex].text) : null;
            return {
              exists: true,
              visible: isVisible(el),
              disabled,
              pointer_events: window.getComputedStyle(el).pointerEvents,
              text: normalizeSafe(el.value !== undefined && el.value !== '' ? el.value : (el.innerText || el.textContent || '')),
              class: normalizeSafe(el.className || ''),
              value: el.value !== undefined ? sanitizeValue(el.value) : null,
              min: el.getAttribute('min') || '',
              max: el.getAttribute('max') || '',
              selected_text: selectedText,
              aria_label: el.getAttribute('aria-label') || '',
              aria_selected: getAriaBool('aria-selected'),
              aria_expanded: getAriaBool('aria-expanded'),
              aria_pressed: getAriaBool('aria-pressed'),
              aria_checked: typeof el.checked !== 'undefined' ? !!el.checked : getAriaBool('aria-checked'),
            };
          };
          const observed = {};
          for (const [id, sel] of Object.entries(observedSelectors || {})) {
            observed[id] = getTargetState(sel);
          }
          const interactiveSelectors = 'button, input, select, textarea, a[href], [onclick], [contenteditable="true"], [tabindex]:not([tabindex="-1"]), [role="button"], [role="tab"], [role="checkbox"], [role="switch"], [role="radio"], [role="option"], [role="menuitem"], [role="slider"]';
          const trackedEvents = new Set(['click', 'mousedown', 'mouseup', 'dblclick', 'contextmenu', 'pointerdown', 'pointerup']);
          const iElements = [];
          try {
            const baseElements = Array.from(document.querySelectorAll(interactiveSelectors));
            const trackedElements = Array.from(document.querySelectorAll('*')).filter((el) => (
              el.__trackedEvents && Array.from(el.__trackedEvents).some((type) => trackedEvents.has(type))
            ));
            const els = Array.from(new Set([...baseElements, ...trackedElements]));
            for (let i = 0; i < els.length && iElements.length < 100; i++) {
              const el = els[i];
              if (!el || !isVisible(el)) continue;
              const getAriaBool = (attr) => {
                const v = el.getAttribute(attr);
                return v === null ? null : v === 'true';
              };
              const ariaDisabled = el.getAttribute('aria-disabled') === 'true';
              const selectedText = el.tagName === 'SELECT' && el.selectedIndex >= 0
                ? normalize(el.options[el.selectedIndex].text) : null;
              iElements.push({
                tag: (el.tagName || '').toLowerCase(),
                id: el.id || '',
                class: normalizeSafe(el.className || ''),
                text: normalizeSafe(el.value !== undefined && el.value !== '' ? el.value : (el.innerText || el.textContent || '')),
                visible: true,
                disabled: typeof el.disabled !== 'undefined' ? !!el.disabled : ariaDisabled,
                pointer_events: window.getComputedStyle(el).pointerEvents,
                value: el.value !== undefined ? sanitizeValue(el.value) : null,
                min: el.getAttribute('min') || '',
                max: el.getAttribute('max') || '',
                selected_text: selectedText,
                aria_label: el.getAttribute('aria-label') || '',
                aria_selected: getAriaBool('aria-selected'),
                aria_expanded: getAriaBool('aria-expanded'),
                aria_pressed: getAriaBool('aria-pressed'),
                aria_checked: typeof el.checked !== 'undefined' ? !!el.checked : getAriaBool('aria-checked'),
              });
            }
          } catch (_) {}
          return {
            page_text: collectPageTextWithVisibility(),
            target: getTargetState(targetSelector),
            observed,
            interactive_elements: iElements
          };
        }
        """,
        {"targetSelector": target_selector, "observedSelectors": observed_selectors or {}},
    )


def synthesize_evidence(before_snap: dict, after_snap: dict, target_selector: str | None) -> dict[str, Any]:
    """Build a dom-log-shaped evidence dict from two raw snapshots.

    Used when the DOM monitor is wiped (e.g. page refresh) and we need to
    reconstruct before/after evidence so [AFTER] assertions and observed
    element assertions can still be evaluated. `events` is left empty -
    [CHANGE] assertions on such transitions should not be written.
    """
    before_target = (before_snap or {}).get("target")
    after_target = (after_snap or {}).get("target")
    final_target_transition = None
    if before_target or after_target:
        all_keys = set((before_target or {}).keys()) | set((after_target or {}).keys())
        changed = {k: (before_target or {}).get(k) != (after_target or {}).get(k) for k in all_keys} if (before_target and after_target) else None
        final_target_transition = {"before": before_target, "after": after_target, "changed": changed}

    observed_elements = {}
    before_observed = (before_snap or {}).get("observed") or {}
    after_observed = (after_snap or {}).get("observed") or {}
    for aid in set(before_observed) | set(after_observed):
        b = before_observed.get(aid)
        a = after_observed.get(aid)
        all_keys = set((b or {}).keys()) | set((a or {}).keys())
        changed = {k: (b or {}).get(k) != (a or {}).get(k) for k in all_keys} if (b and a) else None
        observed_elements[aid] = {"before": b, "after": a, "changed": changed}

    return {
        "target_selector": target_selector,
        "initial_snapshot": {
            "page_text": (before_snap or {}).get("page_text", ""),
            "target": before_target,
            "interactive_elements": (before_snap or {}).get("interactive_elements"),
        },
        "events": [],
        "final_snapshot": {
            "page_text": (after_snap or {}).get("page_text", ""),
            "target": after_target,
            "interactive_elements": (after_snap or {}).get("interactive_elements"),
        },
        "final_target_transition": final_target_transition,
        "observed_elements": observed_elements,
        "synthesized_from_refresh": True,
    }



def summarize_dom_log(dom_log: dict[str, Any]) -> dict[str, Any]:
    events = dom_log.get("events") or []
    added_node_texts = []
    for event in events:
        if not isinstance(event, dict):
            continue
        for node in event.get("added_nodes") or []:
            if not isinstance(node, dict):
                continue
            text = _normalize_text(node.get("text"))
            if text:
                added_node_texts.append(text)

    deduped_texts = []
    seen = set()
    for text in added_node_texts:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_texts.append(text)

    final_snapshot = dom_log.get("final_snapshot") or {}
    final_target = final_snapshot.get("target") or {}

    return {
        "event_count": len(events),
        "added_node_texts": deduped_texts[:20],
        "final_page_text_excerpt": _normalize_text(final_snapshot.get("page_text"))[:300],
        "final_target": {
            "exists": bool(final_target.get("exists", False)),
            "visible": bool(final_target.get("visible", False)),
            "disabled": bool(final_target.get("disabled", False)),
            "text": _normalize_text(final_target.get("text"))[:200],
        },
    }
