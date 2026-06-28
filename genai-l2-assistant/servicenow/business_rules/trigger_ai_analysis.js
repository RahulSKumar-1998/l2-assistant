/**
 * ══════════════════════════════════════════════════════════════════════════════
 * GenAI L2 Support Assistant — ServiceNow Business Rule
 * "Trigger AI Analysis on Incident Create / Assignment Change"
 * ══════════════════════════════════════════════════════════════════════════════
 *
 * DEPLOYMENT INSTRUCTIONS:
 * ────────────────────────
 * 1. Navigate to System Definition > Business Rules (sys_script.do).
 *
 * 2. Create a new Business Rule with these settings:
 *      Name:       GenAI – Trigger AI Analysis
 *      Table:      incident [incident]
 *      Active:     ✓
 *      Advanced:   ✓
 *
 * 3. Under "When to run":
 *      When:       after
 *      Insert:     ✓
 *      Update:     ✓
 *      Filter Conditions:  (leave blank — we filter in script for flexibility)
 *      OR set condition:   assigned_to CHANGES
 *
 * 4. Paste this script into the "Advanced" script field.
 *
 * 5. Create two System Properties (sys_properties.do):
 *      - x_genai.api_endpoint  = https://your-api-host.example.com/api/v1/webhook/incident
 *      - x_genai.hmac_secret   = <your-256-bit-secret>
 *
 * 6. Ensure the MID Server or outbound HTTP allow list includes
 *    your FastAPI endpoint domain.
 *
 * SECURITY NOTES:
 * ───────────────
 * - The webhook payload is signed with HMAC-SHA256.
 * - The backend verifies the signature in the X-ServiceNow-Signature header.
 * - Never log the HMAC secret or full signature in production.
 *
 * PERFORMANCE NOTES:
 * ──────────────────
 * - This is an "after" business rule so it does NOT block the UI.
 * - The REST call is asynchronous (RESTMessageV2 with setEccCorrelator).
 * - If the external call fails, it is logged but does NOT abort the
 *   incident save — AI analysis is non-critical.
 */

(function executeRule(current, previous) {

  // ── Guard: only fire on insert OR when assigned_to changes ──────────
  var isInsert    = current.operation() === 'insert';
  var isAssignChg = previous && current.assigned_to.changes();

  if (!isInsert && !isAssignChg) {
    return;
  }

  // ── Configuration (from System Properties) ─────────────────────────
  var API_ENDPOINT = gs.getProperty('x_genai.api_endpoint',
    'https://your-api-host.example.com/api/v1/webhook/incident');
  var HMAC_SECRET  = gs.getProperty('x_genai.hmac_secret', '');

  if (!HMAC_SECRET) {
    gs.error('[GenAI] HMAC secret not configured. Set x_genai.hmac_secret in sys_properties.');
    return;
  }

  try {
    // ── Build Payload ─────────────────────────────────────────────────
    var payload = {
      event_type:  isInsert ? 'incident.created' : 'incident.assignment_changed',
      sys_id:      current.getUniqueValue(),
      number:      current.number.toString(),
      short_description: current.short_description.toString(),
      description: current.description.toString(),
      priority:    parseInt(current.priority, 10),
      category:    current.category.toString(),
      subcategory: current.subcategory.toString(),
      assigned_to: current.assigned_to.getDisplayValue(),
      assigned_to_id: current.assigned_to.toString(),
      assignment_group: current.assignment_group.getDisplayValue(),
      cmdb_ci:     current.cmdb_ci.getDisplayValue(),
      state:       current.state.toString(),
      opened_at:   current.opened_at.toString(),
      timestamp:   new GlideDateTime().getDisplayValue()
    };

    var payloadStr = JSON.stringify(payload);

    // ── Compute HMAC-SHA256 Signature ─────────────────────────────────
    var mac = new GlideDigest();
    var signature = mac.generateMACBase64('HmacSHA256', HMAC_SECRET, payloadStr);

    // ── Send HTTP POST ────────────────────────────────────────────────
    var request = new sn_ws.RESTMessageV2();
    request.setEndpoint(API_ENDPOINT);
    request.setHttpMethod('POST');
    request.setRequestHeader('Content-Type', 'application/json');
    request.setRequestHeader('X-ServiceNow-Signature', 'sha256=' + signature);
    request.setRequestHeader('X-ServiceNow-Instance', gs.getProperty('instance_name'));
    request.setRequestBody(payloadStr);

    // Execute asynchronously via ECC queue (non-blocking)
    request.setEccParameter('skip_sensor', 'true');
    var response = request.executeAsync();

    // Store the ECC correlation ID for troubleshooting
    var eccCorrelator = response.getEccCorrelator();
    gs.info('[GenAI] AI analysis triggered for ' + current.number +
            ' (event=' + payload.event_type +
            ', ecc=' + eccCorrelator + ')');

  } catch (ex) {
    // ── Error Handling ────────────────────────────────────────────────
    // Log the error but NEVER abort the incident save.
    // AI analysis is supplementary — the engineer can always work without it.
    gs.error('[GenAI] Failed to trigger AI analysis for ' + current.number +
             ': ' + ex.getMessage());

    // Optional: create an event for monitoring dashboards
    gs.eventQueue(
      'x_genai.analysis_failed',
      current,
      ex.getMessage(),
      current.number.toString()
    );
  }

})(current, previous);
