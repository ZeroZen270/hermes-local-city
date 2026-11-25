# Wiring spec — real character models for the NPCs

For the follow-up agent replacing the NPC capsules in `webworld/demo.html`
with the vendored KayKit models. Everything below was measured from the
actual files on 2026-11-24 (GLB header + JSON chunk parsed, skins and
animation accessors read, bind-pose bounds computed). Do not edit this
file's facts without re-measuring.

## Files

| File | Bytes | Body | Notes |
|---|---|---|---|
| `Knight.glb` | 3,659,532 | armored knight, cape, helmet | sha256 `60428e3abc09ba83e595d256e3af8c5c976b46cdae599f0802fc82b4a3445168` (matches the hash pinned by an independent downstream project fetching the same upstream commit) |
| `Barbarian.glb` | 3,613,268 | bare-armed warrior, hat, cape | |
| `Mage.glb` | 3,589,240 | robed mage, tall pointy hat | |
| `Rogue.glb` | 3,616,284 | hooded-down rogue | |
| `Rogue_Hooded.glb` | 3,597,652 | same rogue, hood up, different head node | |
| `*_texture.png` ×4 | ~15 KB each | gradient atlas textures | **Not loaded at runtime** — every GLB embeds its texture (`images[0]` has a `bufferView`, no external URI) and its geometry buffer. The PNGs are reference copies for tint authoring only. |
| `LICENSE-KayKit-Adventurers.txt` | 891 | CC0 license text from the source repo | |

Source: KayKit-Game-Assets/KayKit-Character-Pack-Adventures-1.0 on
GitHub, pinned commit `672074b73ba276876a19e8816ecdc5241817ab47`,
path `addons/kaykit_character_pack_adventures/Characters/gltf/`.
License: **CC0** (verified in the fetched LICENSE.txt). All five GLBs are
glTF 2.0, fully self-contained, 1 skin × 41 joints each, and share an
identical 76-clip animation library (same rig family — any clip plays on
any body).

## What exists today (demo.html, don't fight it)

- NPCs render as **one `InstancedMesh`** of `CapsuleGeometry(0.36, 1.05)`
  (total height ≈ 1.77 m), instance center at `y = 0.95`, per-instance
  color from `ROLE_COLORS`, plus a walk bob. A second InstancedMesh draws
  the tier rings; name labels are canvas sprites at `y = 2.35`.
- Skinned models **cannot** join that InstancedMesh. Render each NPC as
  its own cloned scene graph (below) and keep rings + labels exactly as
  they are — they read only `npc.x / npc.z / npc.heading / npc.tier`,
  which do not change.
- The loader is three.js 0.160.0 via the unpkg importmap already in the
  page; `GLTFLoader` is already imported and used for city models.

## Loading & cloning

```js
import { SkeletonUtils } from 'three/addons/utils/SkeletonUtils.js';
// per body: load once, keep { scene, animations } as the template
// per NPC:  const root = SkeletonUtils.clone(template.scene);
//           const mixer = new THREE.AnimationMixer(root);
```

- Load the five templates lazily *after* the city is up; never block the
  first frame on ~18 MB of characters.
- `SkeletonUtils.clone` shares geometry and materials with the template.
  That is what you want for geometry; for per-NPC tinting, clone the
  materials per NPC (54 material sets is cheap).

## Scale & placement (measured)

Bind pose stands with **feet at Y = 0.00** for all five models — no
ground offset needed; place the group at `(npc.x, 0, npc.z)`,
`rotation.y = npc.heading`.

Body-only bind-pose heights (held props excluded; includes hats/helmets):

| Model | Height | Per-model scale to 1.77 m |
|---|---|---|
| Knight | 2.47 | 0.718 |
| Barbarian | 2.40 | 0.738 |
| Mage | 3.00 (mostly hat) | 0.589 |
| Rogue | 2.19 | 0.809 |
| Rogue_Hooded | 2.25 | 0.786 |

**Recommended: one uniform scale of 0.72 for all five.** They share a
design scale; bodies land ≈ 1.58–1.78 m and hats/helmets rise above,
matching the capsule's 1.77 m presence without shrinking the Mage's body
to fit his hat (the per-model column does that — avoid it). Add a
per-NPC height jitter of ±5 % via `group.scale` if more variety is
wanted. Sanity-check at runtime with `new THREE.Box3().setFromObject(root)`
after the first mixer update.

## Animation mapping (clip names verified identical in all 5 GLBs)

| NPC state (from the movement system) | Clip | Loop | Notes |
|---|---|---|---|
| Dwell / idle at an anchor | `Idle` (1.07 s) | yes | alternate per NPC: `Unarmed_Idle` |
| Engaged in chat | `Idle` | yes | + face the player: heading = atan2(player − npc); this replaces the capsule era's "stop + turn" — the model must actually stop walking, so also kill the walk clip (crossfade below) |
| Travel between anchors | `Walking_A` (1.07 s) | yes | per-NPC variant: `Walking_B`, `Walking_C` (1.6 s) |
| Hurry / long trip (optional) | `Running_A` (0.8 s) | yes | |
| Seated dwell (bench anchors) | `Sit_Chair_Idle` (3.6 s) | yes | enter via `Sit_Chair_Down` (0.8 s) once; `Sit_Floor_Idle` (4 s) for ground spots |
| One-shots (optional flavor) | `Cheer` (greet), `Interact`, `PickUp`, `Use_Item` | no | there is **no** `Wave` clip in this pack — use `Cheer` |

Rules:
- One `AnimationMixer` per NPC; `crossFadeTo(next, 0.25)` on every state
  change — no hard cuts, and never leave a walk clip running under a
  chat (that was the original "won't stop to chat" bug in visual form).
- Desync the crowd: `action.time = (hash(npc.name) % 100) / 100 * duration`
  at start, and per-NPC `action.timeScale = 0.9–1.1`.
- Dwell state comes from the planner (dwell → idle/sit, travel → walk);
  do not infer it from speed after the fact.

## Making 54 NPCs not look like 5 clones

1. **Body assignment by role** (suggestion): guard → Knight,
   mechanic / vendor → Barbarian, medic / elder → Mage,
   courier / student → Rogue, artist / shopkeeper / townsperson →
   Rogue_Hooded. Rotate within a role so repeats aren't adjacent.
2. **Hide the weapons.** Held props are separate named nodes; set
   `visible = false` on them at clone time for a civilian crowd:
   - Knight: `1H_Sword`, `1H_Sword_Offhand`, `2H_Sword`, `Badge_Shield`, `Rectangle_Shield`, `Round_Shield`, `Spike_Shield`
   - Barbarian: `1H_Axe`, `1H_Axe_Offhand`, `2H_Axe`, `Barbarian_Round_Shield` (keep `Mug` for vendors if you like)
   - Mage: `1H_Wand`, `2H_Staff`, `Spellbook`, `Spellbook_open`
   - Rogue / Rogue_Hooded: `Knife`, `Knife_Offhand`, `1H_Crossbow`, `2H_Crossbow`
   Keep capes, hats, helmets — they're body nodes and carry the variety.
3. **Tint.** The atlas is a gradient palette, so `material.color`
   multiplies the whole look. Clone materials per NPC and apply a subtle
   tint derived from the existing `ROLE_COLORS` (keep saturation low,
   ±15 % lightness) so role color-coding survives the capsule swap.
4. **Walk variant + rate + height jitter** per NPC, as above.

## Capsule fallback rule (keep it)

- Character loading is progressive and per-body: keep the existing
  capsule InstancedMesh as the fallback layer. An NPC renders as a
  capsule until its body template has loaded and its clone exists; if a
  GLB fails to load (network, file://, the hosted Space copy), that body
  never swaps in and its NPCs simply stay capsules. Track it per NPC
  (`npc.modelRoot == null` → capsule instance visible).
- Rings, labels, tiers, chat, and god mode must behave identically in
  both render paths — they are driven by NPC state, not by the mesh.
- Perf valve if 54 skinned meshes ever hurt: DORMANT-tier NPCs may stay
  capsules and swap to models on entering AMBIENT. Optional; measure
  first.

## Player (optional)

The player is a separate blue capsule (`CapsuleGeometry(0.38, 1.1)`).
Swapping it for a sixth clone (e.g. Rogue with a strong blue tint) is a
drop-in of the same machinery; not required for the NPC task.
