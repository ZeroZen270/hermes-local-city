# Asset credits — web world

All vendored assets are **CC0 (Creative Commons Zero)** — free for personal
and commercial use, no attribution legally required. We credit anyway.

| Pack | Source | License | What was vendored |
|---|---|---|---|
| Kenney City Kit (Commercial) v2.1 | https://kenney.nl/assets/city-kit-commercial | CC0 (pack License.txt) | `buildings/` — 8 buildings + 3 skyscrapers (GLB) |
| Kenney City Kit (Suburban) | https://kenney.nl/assets/city-kit-suburban | CC0 (pack License.txt) | `houses/` — 6 houses; `props/` — trees, planter |
| Kenney City Kit (Industrial) | https://kenney.nl/assets/city-kit-industrial | CC0 (pack License.txt) | `industrial/` — 4 buildings, chimney, tank, 2 shipping containers, water tower, solar panel |
| Kenney City Kit (Roads) | https://kenney.nl/assets/city-kit-roads | CC0 (pack License.txt) | `roads/` — road pieces, crossing, traffic light, street light, barrier |
| Kenney Car Kit | https://kenney.nl/assets/car-kit | CC0 (pack License.txt) | `vehicles/` — sedan, hatchback, delivery van, police, firetruck, garbage truck, ambulance, race car |
| KayKit City Builder Bits 1.0 (Kay Lousberg) | https://kaylousberg.itch.io/kaykit-city-builder-bits (fetched via the official KayKit-Game-Assets GitHub mirror) | CC0 (pack LICENSE.txt, verified in the download) | `kaykit/` — 4 buildings, road pieces, traffic light, street light, bench, water tower (glTF + .bin + `citybits_texture.png`) |

| KayKit Character Pack: Adventurers 1.0 (Kay Lousberg) | https://github.com/KayKit-Game-Assets/KayKit-Character-Pack-Adventures-1.0 (official KayKit-Game-Assets GitHub mirror; fetched at pinned commit `672074b73ba276876a19e8816ecdc5241817ab47`) — pack page: https://kaylousberg.itch.io/kaykit-adventurers | CC0 (pack LICENSE.txt, fetched from the repo and kept at `characters/LICENSE-KayKit-Adventurers.txt`) | `characters/` — 5 rigged, animated character GLBs (Knight, Barbarian, Mage, Rogue, Rogue_Hooded; 76 animation clips each incl. Idle / Walking / Running / Sit) + the 4 gradient-atlas texture PNGs (reference copies; the GLBs embed their textures and are self-contained) |

**Quaternius animated characters — still not vendored.** Attempted
2026-11-23 and again 2026-11-24: poly.pizza's API requires an API key,
quaternius.com exposes no scriptable download, and the one GitHub mirror
documenting CC0 Quaternius character GLBs (aeytechnologiesllc/sunrise-farm)
returns 404 from the GitHub API. NPC bodies are covered by the KayKit
Adventurers pack above instead; see `characters/WIRING.md` for how they
get wired into the demo.

**Kenney texture atlases (added 2026-11-24, the "buildings are boxes" fix):**
every Kenney GLB above references its pack's shared colour atlas as the
EXTERNAL file `Textures/colormap.png` relative to the model — the GLBs are
not self-contained. The original vendoring copied only the `.glb` files, so
every Kenney model failed to load in the browser and the demo silently
substituted plain fallback boxes (the KayKit characters, whose textures are
embedded, were unaffected). Each pack's real `colormap.png` (512×512) is now
vendored at `<pack folder>/Textures/colormap.png`: Commercial (11,002 B),
Suburban (11,784 B; also used for `props/`, which are Suburban pieces),
Industrial (11,107 B), Roads (11,428 B), Car Kit (12,371 B, SHA-256
f3622a03… verified against a published hash for the identical sedan GLB).
All are part of the same CC0 packs above.

Three.js (demo renderer) is loaded from CDN at runtime, MIT licensed,
https://threejs.org — it is not vendored into this folder.
