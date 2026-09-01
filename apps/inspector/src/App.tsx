import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import {
  createProfile as apiCreateProfile,
  createSession,
  getConfig,
  getCost,
  getHistory,
  listProfiles,
  listSessions,
  runTurn,
  setMode as apiSetMode,
  setProfile as apiSetProfile,
} from "./api";
import ChatPanel from "./components/ChatPanel";
import LoadingScreen, { LOADING_FADE_MS } from "./components/LoadingScreen";
import OperatorSelect from "./components/OperatorSelect";
import ResizeHandles from "./components/ResizeHandles";
import SessionSidebar from "./components/SessionSidebar";
import SettingsPanel from "./components/SettingsPanel";
import TitleBar from "./components/TitleBar";
import TracePanel from "./components/TracePanel";
import type { AttentionMode, RoutingTrace, UiTier } from "./types";

const SCOPE_STORAGE_KEY = "lulu.scope";
const KNOWN_SCOPES_STORAGE_KEY = "lulu.knownScopes";
const UI_TIER_STORAGE_KEY = "lulu.uiTier";

// localStorage can throw (private window, blocked site data) or just be
// unavailable -- these are pure conveniences (remembered scope/profile
// list), never load-bearing, so every access is wrapped rather than
// letting a blocked-storage environment take the whole app down.
function readStorage<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeStorage(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // best-effort; see readStorage's comment
  }
}

export default function App() {
  const queryClient = useQueryClient();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const configQuery = useQuery({ queryKey: ["config"], queryFn: getConfig });

  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [trace, setTrace] = useState<RoutingTrace | null>(null);
  const [hasTurnRun, setHasTurnRun] = useState(false);
  const [mode, setModeState] = useState<AttentionMode | null>(null);
  const [profile, setProfileState] = useState<string | null>(null);

  // Creating AND resuming a session are the same backend call
  // (server.py's create_session resumes if given an id) -- modeled as a
  // mutation, not a query, because "make this the active session" is an
  // action (new session button, clicking a sidebar entry), not something
  // whose cache identity should be keyed by an id that doesn't exist yet
  // for the "new" case.
  const sessionMutation = useMutation({
    mutationFn: (id?: string) => createSession(id),
    onSuccess: (data) => {
      setSessionId(data.session_id);
      setModeState(data.attention_mode);
      setProfileState(data.profile);
      setTrace(null);
      setHasTurnRun(false);
      queryClient.setQueryData(["history", data.session_id], {
        session_id: data.session_id,
        history: data.history,
        attention_mode: data.attention_mode,
        profile: data.profile,
      });
      queryClient.invalidateQueries({ queryKey: ["sessions-list"] });
    },
  });

  useEffect(() => {
    if (sessionMutation.isIdle) sessionMutation.mutate(undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sessionsListQuery = useQuery({
    queryKey: ["sessions-list"],
    queryFn: listSessions,
    enabled: sidebarOpen,
  });

  const historyQuery = useQuery({
    queryKey: ["history", sessionId],
    queryFn: () => getHistory(sessionId!),
    enabled: !!sessionId,
  });

  const costQuery = useQuery({
    queryKey: ["cost", sessionId],
    queryFn: () => getCost(sessionId!),
    enabled: !!sessionId && hasTurnRun,
    retry: false,
  });

  // Memory scope: a per-message tag, not a session switch -- see
  // ScopeSelector.tsx's module comment for why "profile" maps to this
  // and not to a separate chat history per profile.
  const [scope, setScope] = useState<string | null>(() => readStorage(SCOPE_STORAGE_KEY, null));
  const [knownScopes, setKnownScopes] = useState<string[]>(() => readStorage(KNOWN_SCOPES_STORAGE_KEY, []));
  useEffect(() => writeStorage(SCOPE_STORAGE_KEY, scope), [scope]);
  useEffect(() => writeStorage(KNOWN_SCOPES_STORAGE_KEY, knownScopes), [knownScopes]);

  const addKnownScope = (s: string) => {
    setKnownScopes((prev) => (prev.includes(s) ? prev : [...prev, s]));
  };

  // UI tier: picked once via OperatorSelect (null = never picked, show
  // it), then remembered so returning users skip straight past it.
  // TierSwitcher in the title bar can still change it live at any time
  // -- see types.ts's UiTier comment for why this is purely a frontend
  // display-complexity concept, nothing the backend knows about.
  const [tier, setTier] = useState<UiTier | null>(() => readStorage(UI_TIER_STORAGE_KEY, null));
  const handleChangeTier = (t: UiTier) => {
    setTier(t);
    writeStorage(UI_TIER_STORAGE_KEY, t);
  };

  const turnMutation = useMutation({
    mutationFn: (prompt: string) => runTurn(sessionId!, prompt, scope ?? undefined),
    onSuccess: (result) => {
      setTrace(result.trace);
      setHasTurnRun(true);
      queryClient.invalidateQueries({ queryKey: ["history", sessionId] });
      queryClient.invalidateQueries({ queryKey: ["cost", sessionId] });
      queryClient.invalidateQueries({ queryKey: ["sessions-list"] });
    },
  });

  const modeMutation = useMutation({
    mutationFn: (m: AttentionMode) => apiSetMode(sessionId!, m),
  });

  const handleChangeMode = (m: AttentionMode) => {
    setModeState(m);
    modeMutation.mutate(m);
  };

  const profilesQuery = useQuery({ queryKey: ["profiles"], queryFn: listProfiles });

  const profileMutation = useMutation({
    mutationFn: (p: string) => apiSetProfile(sessionId!, p),
  });

  const handleChangeProfile = (p: string) => {
    setProfileState(p);
    profileMutation.mutate(p);
  };

  const createProfileMutation = useMutation({
    mutationFn: ({ name, cloneFrom, persona }: { name: string; cloneFrom: string; persona: string }) =>
      apiCreateProfile(name, cloneFrom, persona || undefined),
    onSuccess: (_data, vars) => {
      queryClient.invalidateQueries({ queryKey: ["profiles"] });
      handleChangeProfile(vars.name);
    },
  });

  const handleCreateProfile = (name: string, cloneFrom: string, persona: string) => {
    createProfileMutation.mutate({ name, cloneFrom, persona });
  };

  const handleSelectSession = (id: string) => {
    if (id === sessionId) {
      setSidebarOpen(false);
      return;
    }
    sessionMutation.mutate(id);
    setSidebarOpen(false);
  };

  const handleNewSession = () => {
    sessionMutation.mutate(undefined);
    setSidebarOpen(false);
  };

  // Loading/error gate covers reachability of lulu-server itself, not
  // whether a model API key is configured -- ChatPanel already lets you
  // type and send without one (see its own module comment); a missing
  // key surfaces as a normal error in that turn's response, not a
  // blocking setup wizard here.
  const hasError = configQuery.isError || sessionMutation.isError;
  const isReady = !configQuery.isLoading && !sessionMutation.isIdle && !sessionMutation.isPending && !hasError;

  // The parallax loading screen is decorative -- on localhost, config +
  // session both resolve in well under a second, which isn't enough time
  // to actually see the walk cycle or the grass scroll. MIN_LOADING_MS
  // enforces a floor so it's visible regardless of how fast the backend
  // answers; showLoading then stays true a bit longer still so the exit
  // fade (styles.css's .loading-screen.exiting) has time to play while
  // the app fades in underneath, instead of either screen just vanishing
  // instantly.
  const MIN_LOADING_MS = 4200;
  const [mountedAt] = useState(() => Date.now());
  const [fading, setFading] = useState(false);
  const [showLoading, setShowLoading] = useState(true);

  useEffect(() => {
    if (!isReady || fading || !showLoading) return;
    const remaining = Math.max(0, MIN_LOADING_MS - (Date.now() - mountedAt));
    const timer = setTimeout(() => setFading(true), remaining);
    return () => clearTimeout(timer);
  }, [isReady, fading, showLoading, mountedAt]);

  useEffect(() => {
    if (!fading) return;
    const timer = setTimeout(() => setShowLoading(false), LOADING_FADE_MS);
    return () => clearTimeout(timer);
  }, [fading]);

  const history = historyQuery.data?.history ?? [];

  return (
    <>
      <ResizeHandles />
      {(showLoading || hasError) && (
        <LoadingScreen
          status={hasError ? "error" : "connecting"}
          fading={fading}
          onRetry={
            hasError
              ? () => {
                  configQuery.refetch();
                  sessionMutation.reset();
                  sessionMutation.mutate(undefined);
                }
              : undefined
          }
        />
      )}
      {!showLoading && !hasError && tier === null && <OperatorSelect onSelect={handleChangeTier} />}
      {!showLoading && !hasError && tier !== null && (
        <div className="app">
          <TitleBar
            onToggleSettings={() => setSettingsOpen((v) => !v)}
            onToggleSessions={() => setSidebarOpen((v) => !v)}
            tier={tier}
            onChangeTier={handleChangeTier}
            mode={mode}
            onChangeMode={handleChangeMode}
            scope={scope}
            knownScopes={knownScopes}
            onChangeScope={setScope}
            onAddScope={addKnownScope}
            profile={profile}
            profiles={profilesQuery.data?.profiles ?? ["default"]}
            onLoadProfiles={() => profilesQuery.refetch()}
            onChangeProfile={handleChangeProfile}
            onCreateProfile={handleCreateProfile}
            creatingProfile={createProfileMutation.isPending}
            createProfileError={createProfileMutation.isError ? createProfileMutation.error.message : null}
          />
          {settingsOpen && (
            <SettingsPanel config={configQuery.data ?? null} onClose={() => setSettingsOpen(false)} />
          )}
          <SessionSidebar
            open={sidebarOpen}
            sessions={sessionsListQuery.data?.sessions ?? []}
            activeSessionId={sessionId}
            onSelect={handleSelectSession}
            onNewSession={handleNewSession}
            onClose={() => setSidebarOpen(false)}
          />
          <div className="layout">
            <ChatPanel
              history={history}
              pending={turnMutation.isPending || !sessionId}
              error={turnMutation.isError ? turnMutation.error : null}
              onSend={(prompt) => {
                turnMutation.reset();
                turnMutation.mutate(prompt);
              }}
            />
            <TracePanel trace={trace} cost={costQuery.data ?? null} />
          </div>
        </div>
      )}
    </>
  );
}
