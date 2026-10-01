// The players of the iOS app. The ids are shared with ios/DecisionPacman/Players.swift,
// the `decisionpacman://control?model=` route, and the page's `?model=` parameter.

export type PlayerId = "finetuned" | "phi4-mini" | "jev";

export interface PlayerInfo {
  label: string;
  where: "on device" | "cloud";
}

export const PLAYERS: Record<PlayerId, PlayerInfo> = {
  finetuned: { label: "fine-tuned 0.8B", where: "on device" },
  "phi4-mini": { label: "phi4-mini 3.8B", where: "on device" },
  jev: { label: "Jev", where: "cloud" },
};

export const DEFAULT_PLAYER: PlayerId = "finetuned";

export function isPlayerId(value: unknown): value is PlayerId {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(PLAYERS, value);
}

export function playerLabel(id: PlayerId): string {
  return `${PLAYERS[id].label} (${PLAYERS[id].where})`;
}
