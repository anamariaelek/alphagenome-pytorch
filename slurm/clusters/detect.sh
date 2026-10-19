# Sourced (not executed). Sets AG_CLUSTER to helix | uc3 | horeka | sasselab3.
#
# Preferred: export AG_CLUSTER=<name> in each machine's ~/.bashrc. Falls back to hostname
# patterns, and errors out rather than guessing if neither matches.

if [[ -z "${AG_CLUSTER:-}" ]]; then
    case "$(hostname -s)" in
        helix*|hlx*) AG_CLUSTER=helix ;;
        uc3*)        AG_CLUSTER=uc3 ;;
        hkn*)        AG_CLUSTER=horeka ;;
        sasselab3*)  AG_CLUSTER=sasselab3 ;;
        *)
            echo "ERROR: cannot detect cluster from hostname '$(hostname -s)'." >&2
            echo "       Set AG_CLUSTER=helix|uc3|horeka|sasselab3 (e.g. in ~/.bashrc)." >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
fi
export AG_CLUSTER
