// Fracture care guide: general guidance, summarised from the NHS and Mayo Clinic
// pages listed in CARE_SOURCES. It is information, not advice for a specific injury.
// "general" applies to every body part; each region adds or replaces items.

export const EMERGENCY_NUMBERS = "112 in India and the EU, 999 in the UK, 911 in the US";

export const CARE_SOURCES = [
  { title: "Fractures: first aid (Mayo Clinic)", url: "https://www.mayoclinic.org/first-aid/first-aid-fractures/basics/art-20056641" },
  { title: "Broken arm or wrist (NHS)", url: "https://www.nhs.uk/conditions/broken-arm-or-wrist/" },
  { title: "Broken ankle (NHS)", url: "https://www.nhs.uk/conditions/broken-ankle/" },
  { title: "Hip fracture (NHS)", url: "https://www.nhs.uk/conditions/hip-fracture/" },
  { title: "Wrist fracture leaflet (Gloucestershire Hospitals NHS)", url: "https://gloshospitals.nhs.uk/your-visit/patient-information-leaflets/wrist-fracture-ghpi1508_03_19" },
];

export const REGIONS = [
  { id: "general", label: "Any bone" },
  { id: "hand", label: "Hand and wrist" },
  { id: "arm", label: "Arm, elbow and shoulder" },
  { id: "leg", label: "Leg, ankle and foot" },
  { id: "hip", label: "Hip and pelvis" },
];

const GENERAL = {
  emergency: [
    "The bone has come through the skin, or there is a deep wound",
    "Bleeding is heavy",
    "The limb or joint looks bent, twisted or out of shape",
    "Fingers or toes beyond the injury are numb, tingling, pale or blue",
    "A break in the head, neck or back is possible, or the injury came from a major accident",
    "The person is faint, pale and breathing fast (possible shock)",
  ],
  firstAid: [
    "Keep the injured part still and supported. Don't try to straighten it or push a bone back in.",
    "Stop any bleeding by pressing a clean cloth around the wound, not on the bone.",
    "Cool it: an ice pack or frozen peas wrapped in a towel, up to 20 minutes every 2 to 3 hours. Never put ice straight on skin.",
    "Remove rings, watches and tight jewellery near the injury before it swells.",
    "Don't eat or drink until a doctor has seen it, in case it needs surgery.",
    "Don't drive yourself. Ask someone to take you, or call an ambulance.",
  ],
  treatment: [
    "An X-ray confirms the break and shows how the bone pieces line up.",
    "Many breaks heal in a splint, cast or support boot that keeps the bone still.",
    "If the pieces have moved, a doctor may realign them after pain relief or anaesthetic (a “reduction”).",
    "Some breaks need surgery to hold the bone with plates, screws, pins or rods.",
    "Follow-up X-rays check that the bone is healing in the right position.",
  ],
  healing: "Most adult breaks take about 6 to 12 weeks to heal, and children's bones usually heal faster. Stiffness and weakness can last for weeks to months after the cast comes off; physiotherapy and the exercises you are given help most.",
  dos: [
    "Keep the injured part raised above heart level when you can, to reduce swelling",
    "Gently move the joints the cast doesn't cover, as often as advised",
    "Take pain relief as advised by your doctor or pharmacist",
    "Eat well, with enough protein, calcium and vitamin D, and avoid smoking, which slows bone healing",
    "Go to every follow-up appointment",
  ],
  donts: [
    "Don't let a plaster cast get wet; pharmacies sell waterproof covers",
    "Don't push anything inside the cast to scratch",
    "Don't trim or remove the cast yourself",
    "Don't lift heavy things, drive or play contact sport until you are told it is safe",
  ],
  warning: [
    "Pain that keeps getting worse",
    "Numbness, burning, or pins and needles below the injury",
    "Fingers or toes that swell, or turn blue or white",
    "A cast that feels too tight, loose, cracked, or rubs",
    "A bad smell or discharge from under the cast, or a fever and chills",
  ],
};

const REGION_EXTRAS = {
  hand: {
    firstAid: ["Support the arm in a sling made from a towel or scarf, with the hand higher than the elbow."],
    healing: "A wrist cast or splint is usually worn for up to about 6 weeks; full strength can take a few months to return.",
    dos: [
      "Make a fist and straighten your fingers often, and keep your elbow and shoulder moving so they don't stiffen",
      "Use the hand for light tasks like dressing and eating if your team says so",
      "Rest the hand on a pillow at night",
    ],
  },
  arm: {
    firstAid: ["Support the arm close to the body in a sling, in the position that hurts least."],
    healing: "Arm and elbow breaks usually take 6 to 12 weeks in adults. Shoulder (upper arm) breaks often start gentle movement early, as guided by a physiotherapist, to stop the joint stiffening.",
    dos: ["Keep the fingers, wrist and any free joints moving", "Wear the sling the way you were shown, and only take it off when advised"],
  },
  leg: {
    firstAid: ["Don't put weight on the leg. Keep it raised on cushions while waiting.", "Bandage loosely for support only if the ankle isn't at an odd angle."],
    healing: "A broken ankle usually takes about 6 to 8 weeks, sometimes longer. Your team tells you how much weight you can put on it; you may get crutches or a frame.",
    dos: ["Wiggle your toes and bend your knee often while in a boot or cast", "Keep the leg raised above hip level when sitting, especially in the first days"],
    donts: ["Don't walk on the cast or boot unless you have been told you can"],
  },
  hip: {
    emergencyNote: "A suspected broken hip always needs emergency care: call an ambulance. Signs include hip or groin pain after a fall, not being able to stand or put weight on the leg, and a leg that looks shorter or turned outwards.",
    treatment: [
      "Almost all hip fractures need surgery, usually on the day of admission or the next day.",
      "Depending on the break, surgeons fix the bone with screws, nails or plates, or replace part or all of the hip joint.",
      "Physiotherapy starts soon after surgery to get you standing and walking again.",
    ],
    healing: "Recovery takes weeks to months. Hospital stays are often 1 to 4 weeks, and some people need a walking aid or help at home for a while.",
    dos: [
      "Do the home exercises you are given every day",
      "If you are over 50, ask your doctor about a bone-health (osteoporosis) check and a falls-prevention plan to prevent another break",
    ],
  },
};

/** The guide for one region: general items plus the region's own. */
export function careFor(region) {
  const extra = REGION_EXTRAS[region] || {};
  return {
    emergency: GENERAL.emergency,
    emergencyNote: extra.emergencyNote || null,
    firstAid: [...(extra.firstAid || []), ...GENERAL.firstAid],
    treatment: extra.treatment || GENERAL.treatment,
    healing: extra.healing || GENERAL.healing,
    dos: [...(extra.dos || []), ...GENERAL.dos],
    donts: [...GENERAL.donts, ...(extra.donts || [])],
    warning: GENERAL.warning,
  };
}
