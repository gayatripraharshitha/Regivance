import React from "react";

export class AppErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error:false }; }
  static getDerivedStateFromError() { return { error:true }; }
  componentDidCatch(error) { console.error("Regivance view error", error); }
  render() {
    if (this.state.error) return <main className="route-error"><span>REGIVANCE</span><h1>This view could not load.</h1><p>Your data is safe. Reload the page to recover; if the problem continues, return to the overview.</p><div><button onClick={() => window.location.reload()}>Reload page</button><a href="/">Open overview</a></div></main>;
    return this.props.children;
  }
}
