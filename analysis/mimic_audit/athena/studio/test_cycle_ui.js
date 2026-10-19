// jsdom render test for the cycle-consistency studio page.
// Drives web/cycle.html against a live server (default :8046) via a fetch shim,
// then asserts the overview table, model detail (shift matrix + gate flow),
// how-it-works, claim browser and drill-down all render from real data.
//
//   python3 server.py --port 8046 &        # (test starts/uses this)
//   node test_cycle_ui.js 8046
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const PORT = process.argv[2] || "8046";
const BASE = `http://127.0.0.1:${PORT}`;
const html = fs.readFileSync(path.join(__dirname, "web", "cycle.html"), "utf8");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
function ok(cond, msg) { if (cond) console.log("  ok  " + msg); else { failures++; console.log("  FAIL " + msg); } }

(async () => {
  // inject the fetch shim BEFORE the inline script runs (the IIFE fetches during parse)
  const dom = new JSDOM(html, {
    runScripts: "dangerously", pretendToBeVisual: true,
    beforeParse(window) {
      window.fetch = (u, opts) => fetch(u.startsWith("http") ? u : BASE + u, opts);
    },
  });
  const w = dom.window;

  // wait for the initial /api/cycle/runs render
  for (let i = 0; i < 40 && !w.document.querySelector(".modelrow"); i++) await sleep(100);
  const app = w.document.getElementById("app");

  console.log("[overview]");
  const rows = w.document.querySelectorAll(".modelrow");
  ok(rows.length >= 2, `>=2 model rows (got ${rows.length})`);
  ok(w.document.querySelector(".sbar > div") != null, "stacked verdict bar rendered");
  ok(/Cross-model replication/.test(app.textContent), "overview heading present");
  const nav = w.document.querySelectorAll("#nav button");
  ok(nav.length >= 4, `nav has overview+how+models (got ${nav.length})`);

  console.log("[how it works]");
  [...nav].find((b) => b.dataset.tab === "how").click();
  await sleep(60);
  ok(/three readings/i.test(app.textContent), "how-it-works: three readings section");
  ok(/M_BB/.test(app.textContent) && /D\*/.test(app.textContent), "how-it-works: reading formulae");
  ok(/grounded/.test(app.textContent) && /confabulated/.test(app.textContent), "how-it-works: verdict glossary");

  console.log("[model detail]");
  // click first model tab
  const mtab = [...w.document.querySelectorAll("#nav button")].find((b) => b.dataset.tab.startsWith("m:"));
  const mkey = mtab.dataset.tab.slice(2);
  mtab.click();
  for (let i = 0; i < 40 && !w.document.querySelector(".flow .fnode"); i++) await sleep(100);
  ok(w.document.querySelector("table.shift") != null, "atom->cycle shift matrix rendered");
  ok(w.document.querySelectorAll(".flow .fnode").length >= 4, "gate-flow nodes rendered");
  ok(/grounding estimate/i.test(app.textContent), "grounding-estimate card present");
  ok(/out-of-sample/i.test(app.textContent), "OOS kappa card present");
  ok(/Tier-1 unresolved recovery/.test(app.textContent), "Tier-1 before/after panel present");
  ok(/FP unchanged/.test(app.textContent), "Tier-1 FP-unchanged badge present");
  ok(/before/.test(app.textContent) && /after/.test(app.textContent), "Tier-1 before+after labels");
  ok(w.document.querySelectorAll(".sbar").length >= 3, "Tier-1 before+after stacked bars rendered");
  ok(/Tier-0 . recalibration|Tier-0 .. recalibration|Tier-0/.test(app.textContent), "Tier-0 recalibration panel present");
  ok(/transfers OOS|OOS FP/.test(app.textContent), "Tier-0 FP-budget badge present");
  ok(/Tier-3 band-widening/.test(app.textContent), "Tier-3 band-widening panel present");
  ok(/FP held/.test(app.textContent), "Tier-3 FP-held badge present");
  ok(/variant/.test(app.textContent), "Tier-3 variant badge present");
  ok(/P90/.test(app.textContent) && /P95/.test(app.textContent), "Tier-3 P90/P95 before+after labels");

  console.log("[claim browser + drilldown]");
  for (let i = 0; i < 50 && !w.document.querySelector(".crow"); i++) await sleep(100);
  const crows = w.document.querySelectorAll(".crow");
  ok(crows.length >= 1, `claim rows rendered (got ${crows.length})`);
  ok(w.document.querySelectorAll("#filters button").length === 6, "6 verdict filters (all+5)");
  crows[0].click();
  for (let i = 0; i < 40 && !w.document.querySelector(".drill .readbar"); i++) await sleep(100);
  ok(w.document.querySelector(".drill .readbar") != null, "drilldown readbar (M_BB/X/D*) rendered");
  ok(w.document.querySelector(".drill table.decoys") != null, "drilldown decoy panel rendered");

  console.log(failures ? `\nFAILURES: ${failures}` : "\nALL CYCLE-UI TESTS PASSED");
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
