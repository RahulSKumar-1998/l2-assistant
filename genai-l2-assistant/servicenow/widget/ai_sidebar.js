/**
 * ══════════════════════════════════════════════════════════════════════════════
 * GenAI L2 Support Assistant — ServiceNow AI Sidebar Widget (Client Controller)
 * ══════════════════════════════════════════════════════════════════════════════
 *
 * DEPLOYMENT INSTRUCTIONS (ServiceNow):
 * ──────────────────────────────────────
 * 1. Paste this code into the "Client controller" field of the widget
 *    record in the Widget Designer (sp_widget.do).
 *
 * 2. BACKEND URL: Update API_BASE_URL to point at your FastAPI deployment.
 *    In production, use a ServiceNow Scripted REST Message or MID Server proxy
 *    instead of direct browser-to-API calls to avoid CORS issues.
 *
 * 3. The controller reads the incident sys_id from:
 *       - Widget option:  c.options.sysId
 *       - g_form fallback: g_form.getUniqueValue()
 *    Make sure one of these is available when the widget loads.
 *
 * 4. Engineer identity is pulled from NOW.user.userID (ServiceNow global).
 *    For local testing, it falls back to "test-engineer-001".
 *
 * ARCHITECTURE NOTES:
 * ───────────────────
 * - On load, the controller POST /api/v1/incidents/analyze with the ticket
 *   sys_id to trigger asynchronous RAG analysis.
 * - It then polls GET /api/v1/recommendations/{id} every 3 seconds until
 *   the backend returns status: "complete".
 * - Chat messages go through POST /api/v1/chat.
 * - Feedback is submitted via POST /api/v1/feedback.
 * - copyToWorkNotes() uses ServiceNow's g_form.setValue to pre-populate
 *   the work_notes journal field on the incident form.
 */

/* global angular, NOW, g_form */

(function () {
  'use strict';

  // ── Configuration ───────────────────────────────────────────────────────
  // In production, route through a ServiceNow Scripted REST Message
  // or MID Server proxy to avoid browser CORS restrictions.
  var API_BASE_URL = 'https://your-api-host.example.com/api/v1';
  var POLL_INTERVAL_MS = 3000;
  var MAX_POLL_ATTEMPTS = 60; // 3 minutes max polling

  angular.module('genai.ai_sidebar', []).controller('AISidebarController', [
    '$scope',
    '$http',
    '$interval',
    '$timeout',
    function ($scope, $http, $interval, $timeout) {
      var c = this;

      // ── State ─────────────────────────────────────────────────────────
      c.isLoading = true;
      c.status = 'analysing';       // 'analysing' | 'ready' | 'low_confidence'
      c.statusLabel = 'Analysing…';
      c.recommendation = null;
      c.errorMessage = null;

      // Triage step tracking
      c.completedSteps = {};
      c.expandedSteps = {};

      // Chat state
      c.chatInput = '';
      c.chatMessages = [];
      c.chatSessionId = null;
      c.suggestedPrompts = [
        'Likely cause?',
        'Show similar incidents',
        'Check runbook',
        'Escalation criteria'
      ];

      // Feedback state
      c.feedbackRating = null;
      c.feedbackComment = '';
      c.feedbackSubmitted = false;

      // Internal
      var _pollHandle = null;
      var _pollCount = 0;
      var _recommendationId = null;

      // ── Resolve Context ───────────────────────────────────────────────
      /**
       * Get the current incident sys_id.
       * Prefer widget option, fall back to g_form on the parent page.
       */
      function getSysId() {
        if (c.options && c.options.sysId) {
          return c.options.sysId;
        }
        // g_form is available on the standard incident form workspace
        if (typeof g_form !== 'undefined' && g_form.getUniqueValue) {
          return g_form.getUniqueValue();
        }
        return null;
      }

      /**
       * Get the current engineer's user ID from ServiceNow globals.
       */
      function getEngineerId() {
        if (typeof NOW !== 'undefined' && NOW.user && NOW.user.userID) {
          return NOW.user.userID;
        }
        return 'test-engineer-001';
      }

      // ── API Helpers ───────────────────────────────────────────────────
      /**
       * Standard headers for FastAPI backend requests.
       * In production, add Authorization: Bearer <token> here.
       */
      function apiHeaders() {
        return {
          'Content-Type': 'application/json',
          'X-Engineer-Id': getEngineerId()
        };
      }

      // ── 1. Trigger Analysis ───────────────────────────────────────────
      /**
       * POST /api/v1/incidents/analyze
       * Kicks off the asynchronous RAG pipeline for the current incident.
       */
      function triggerAnalysis() {
        var sysId = getSysId();
        if (!sysId) {
          c.isLoading = false;
          c.errorMessage = 'No incident sys_id available. Ensure the widget is placed on an incident form.';
          return;
        }

        $http({
          method: 'POST',
          url: API_BASE_URL + '/incidents/analyze',
          headers: apiHeaders(),
          data: { sys_id: sysId }
        }).then(function (response) {
          _recommendationId = response.data.recommendation_id || response.data.id;
          startPolling();
        }).catch(function (err) {
          c.isLoading = false;
          c.errorMessage = 'Failed to trigger analysis: ' + (err.data && err.data.detail || err.statusText || 'Unknown error');
        });
      }

      // ── 2. Poll for Recommendation ────────────────────────────────────
      /**
       * GET /api/v1/recommendations/{id} every POLL_INTERVAL_MS.
       * Stops when status is "complete" or max attempts reached.
       */
      function startPolling() {
        _pollHandle = $interval(function () {
          _pollCount++;

          if (_pollCount > MAX_POLL_ATTEMPTS) {
            $interval.cancel(_pollHandle);
            c.isLoading = false;
            c.errorMessage = 'Analysis timed out. Please retry.';
            return;
          }

          $http({
            method: 'GET',
            url: API_BASE_URL + '/recommendations/' + _recommendationId,
            headers: apiHeaders()
          }).then(function (response) {
            var data = response.data;

            if (data.status === 'complete' || data.root_cause_prediction) {
              $interval.cancel(_pollHandle);
              c.recommendation = data;
              c.isLoading = false;
              updateStatus(data.confidence_score);
            }
          }).catch(function (err) {
            // Non-fatal: keep polling unless it's a 404 (recommendation deleted)
            if (err.status === 404) {
              $interval.cancel(_pollHandle);
              c.isLoading = false;
              c.errorMessage = 'Recommendation not found. It may have expired.';
            }
          });
        }, POLL_INTERVAL_MS);
      }

      /**
       * Update the header status badge based on confidence score.
       */
      function updateStatus(confidence) {
        if (confidence < 0.6) {
          c.status = 'low_confidence';
          c.statusLabel = 'Low confidence';
        } else {
          c.status = 'ready';
          c.statusLabel = 'Ready';
        }
      }

      // ── 3. Copy to Work Notes ─────────────────────────────────────────
      /**
       * Push the resolution draft into ServiceNow's work_notes journal field.
       * Uses g_form.setValue() which is the standard ServiceNow client API.
       */
      c.copyToWorkNotes = function () {
        var text = c.recommendation && c.recommendation.resolution_draft;
        if (!text) { return; }

        // g_form.setValue('work_notes', value) adds a new journal entry
        if (typeof g_form !== 'undefined' && g_form.setValue) {
          g_form.setValue('work_notes', '[AI Recommendation]\n' + text);
        } else {
          // Fallback: copy to clipboard (for local development / simulator)
          c.copyToClipboard(text);
        }
      };

      // ── 4. Chat (Ask AI) ──────────────────────────────────────────────
      /**
       * Send a chat message to the AI backend.
       * POST /api/v1/chat
       * Payload matches ChatRequest model:
       *   { incident_id, message, session_id?, engineer_id }
       */
      c.sendChatMessage = function (message) {
        if (!message || !message.trim()) { return; }

        var userMsg = message.trim();
        c.chatInput = '';

        // Optimistic UI: show user message immediately
        c.chatMessages.push({ role: 'user', content: userMsg });

        $http({
          method: 'POST',
          url: API_BASE_URL + '/chat',
          headers: apiHeaders(),
          data: {
            incident_id: getSysId(),
            message: userMsg,
            session_id: c.chatSessionId,
            engineer_id: getEngineerId()
          }
        }).then(function (response) {
          c.chatSessionId = response.data.session_id;
          c.chatMessages.push({
            role: 'assistant',
            content: response.data.response,
            sources: response.data.sources || []
          });
        }).catch(function (err) {
          c.chatMessages.push({
            role: 'assistant',
            content: 'Sorry, I encountered an error. Please try again.'
          });
        });
      };

      // ── 5. Feedback ───────────────────────────────────────────────────
      /**
       * Submit thumbs up (5) or thumbs down (1) feedback.
       * POST /api/v1/feedback
       * Payload matches FeedbackSubmission model.
       */
      c.submitFeedback = function (rating) {
        c.feedbackRating = rating;
        c.feedbackSubmitted = false;

        var payload = {
          recommendation_id: c.recommendation.id,
          rating: rating,
          engineer_id: getEngineerId(),
          acted_on_steps: Object.keys(c.completedSteps)
            .filter(function (k) { return c.completedSteps[k]; })
            .map(Number)
        };

        $http({
          method: 'POST',
          url: API_BASE_URL + '/feedback',
          headers: apiHeaders(),
          data: payload
        }).then(function () {
          c.feedbackSubmitted = true;
        }).catch(function () {
          // Silently fail — feedback is non-critical
        });
      };

      /**
       * Submit the optional feedback comment.
       * Re-sends feedback with the comment field populated.
       */
      c.submitFeedbackComment = function () {
        if (!c.feedbackComment.trim()) { return; }

        $http({
          method: 'POST',
          url: API_BASE_URL + '/feedback',
          headers: apiHeaders(),
          data: {
            recommendation_id: c.recommendation.id,
            rating: c.feedbackRating,
            comment: c.feedbackComment.trim(),
            engineer_id: getEngineerId(),
            acted_on_steps: Object.keys(c.completedSteps)
              .filter(function (k) { return c.completedSteps[k]; })
              .map(Number)
          }
        }).then(function () {
          c.feedbackSubmitted = true;
        });
      };

      // ── 6. Utility Functions ──────────────────────────────────────────

      /**
       * Toggle rationale visibility for a triage step.
       */
      c.toggleRationale = function (stepNum) {
        c.expandedSteps[stepNum] = !c.expandedSteps[stepNum];
      };

      /**
       * Handle triage step checkbox toggle.
       */
      c.onStepToggle = function (stepNum) {
        // Could be extended to send partial progress to backend
      };

      /**
       * Copy text to clipboard using the Clipboard API.
       */
      c.copyToClipboard = function (text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text);
        } else {
          // Legacy fallback
          var textarea = document.createElement('textarea');
          textarea.value = text;
          textarea.style.position = 'fixed';
          textarea.style.left = '-9999px';
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand('copy');
          document.body.removeChild(textarea);
        }
      };

      /**
       * Format resolution time from minutes to a human-readable string.
       */
      c.formatResolutionTime = function (minutes) {
        if (!minutes) { return ''; }
        if (minutes < 60) { return minutes + ' min'; }
        var hours = Math.floor(minutes / 60);
        var mins = minutes % 60;
        if (hours < 24) { return hours + 'h ' + mins + 'm'; }
        var days = Math.floor(hours / 24);
        hours = hours % 24;
        return days + 'd ' + hours + 'h';
      };

      /**
       * Retry analysis after an error.
       */
      c.retryAnalysis = function () {
        c.errorMessage = null;
        c.isLoading = true;
        c.status = 'analysing';
        c.statusLabel = 'Analysing…';
        _pollCount = 0;
        triggerAnalysis();
      };

      // ── Lifecycle ─────────────────────────────────────────────────────

      // Clean up polling interval when scope is destroyed
      $scope.$on('$destroy', function () {
        if (_pollHandle) {
          $interval.cancel(_pollHandle);
        }
      });

      // Kick off analysis on widget load
      triggerAnalysis();
    }
  ]);
})();
