import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { createSession, getConfig, getCost, getHistory, runTurn } from "./api";
import ChatPanel from "./components/ChatPanel";
import TracePanel from "./components/TracePanel";
import type { RoutingTrace } from "./types";

export default function App() {
  const queryClient = useQueryClient();

  const configQuery = useQuery({ queryKey: ["config"], queryFn: getConfig });
  const sessionQuery = useQuery({
    queryKey: ["session"],
    queryFn: () => createSession(),
    staleTime: Infinity,
  });
  const sessionId = sessionQuery.data?.session_id;

  const [trace, setTrace] = useState<RoutingTrace | null>(null);
  const [hasTurnRun, setHasTurnRun] = useState(false);

  const historyQuery = useQuery({
    queryKey: ["history", sessionId],
    queryFn: () => getHistory(sessionId!),
    enabled: !!sessionId,
    initialData: sessionQuery.data ? { session_id: sessionQuery.data.session_id, history: sessionQuery.data.history } : undefined,
  });

  const costQuery = useQuery({
    queryKey: ["cost", sessionId],
    queryFn: () => getCost(sessionId!),
    enabled: !!sessionId && hasTurnRun,
    retry: false,
  });

  const turnMutation = useMutation({
    mutationFn: (prompt: string) => runTurn(sessionId!, prompt),
    onSuccess: (result) => {
      setTrace(result.trace);
      setHasTurnRun(true);
      queryClient.invalidateQueries({ queryKey: ["history", sessionId] });
      queryClient.invalidateQueries({ queryKey: ["cost", sessionId] });
    },
  });

  const history = historyQuery.data?.history ?? [];

  return (
    <div className="app">
      <div className="header">
        <span className="header-title">Lulu Inspector</span>
        {configQuery.data && (
          <span className="header-meta">
            <span>provider={configQuery.data.provider}</span>
            <span>model={configQuery.data.model}</span>
            <span>mode={configQuery.data.attention_mode}</span>
          </span>
        )}
      </div>
      <div className="layout">
        <ChatPanel
          history={history}
          pending={turnMutation.isPending || !sessionId}
          onSend={(prompt) => turnMutation.mutate(prompt)}
        />
        <TracePanel trace={trace} cost={costQuery.data ?? null} />
      </div>
    </div>
  );
}
