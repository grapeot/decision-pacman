// Where the player's 30-second score lands among the models' scores on the same seed.

export interface Entry {
  id: string;
  pellets: number;
  you?: boolean;
}

/**
 * Sorts the player in among the others, best first. A tie goes to the player.
 * Returns the ordered list and the player's 1-based rank.
 */
export function rankPlayer(you: number, others: readonly Entry[]): { order: Entry[]; rank: number } {
  const order: Entry[] = others.map((e) => ({ ...e })).sort((a, b) => b.pellets - a.pellets);
  let at = order.findIndex((e) => you >= e.pellets);
  if (at < 0) at = order.length;
  order.splice(at, 0, { id: "you", pellets: you, you: true });
  return { order, rank: at + 1 };
}
