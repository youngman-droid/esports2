#!/bin/sh
# Back up the live Postgres/TimescaleDB database, data/, and the code to a
# mounted share (the Windows D: drive over Tailscale).  Two phases so the slow
# pg_dump can run before the share is mounted:
#   sh scripts/backup_to_windows.sh dump              # pg_dump -> $STAGE (local)
#   sh scripts/backup_to_windows.sh sync /Volumes     # copy latest staged dump + data/ + code -> <dest>/$BACKUP_SUBDIR, verify
#   sh scripts/backup_to_windows.sh all  /Volumes     # both (-> /Volumes/esports2-backup)
# Env: LOL_TICKER_DSN (postgresql://localhost:5432/league), PGBIN (/opt/homebrew/opt/postgresql@18/bin),
#      STAGE (~/.cache/esports2-backup), BACKUP_SUBDIR (esports2-backup), VERIFY=full (re-read the copied
#      dump from the share and compare sha256; default checks size + archive TOC only), KEEP_STAGE=1 (keep local dump),\n#      TAR_MIN_FILES (500: second-level data/ folders with more files than this are shipped as one tar).
set -eu
cd "$(dirname "$0")/.."
REPO=$(pwd -P)
DSN="${LOL_TICKER_DSN:-postgresql://localhost:5432/league}"
PGBIN="${PGBIN:-/opt/homebrew/opt/postgresql@18/bin}"
STAGE="${STAGE:-$HOME/.cache/esports2-backup}"
SUB="${BACKUP_SUBDIR:-esports2-backup}"
log() { printf '%s backup: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

dump() {
  mkdir -p "$STAGE"
  stamp=$(date -u +%Y%m%dT%H%M%SZ)            # no colons: the files land on NTFS
  out="$STAGE/league_$stamp.pgdump"
  log "pg_dump -Fc --compress=zstd:9 -> $out"
  "$PGBIN/pg_dump" -Fc --compress=zstd:9 "$DSN" > "$out.partial"
  mv "$out.partial" "$out"
  "$PGBIN/pg_restore" --list "$out" > /dev/null || die "archive TOC unreadable"
  "$PGBIN/pg_dumpall" --globals-only -d "$DSN" > "$STAGE/globals_$stamp.sql"
  {
    echo "created_utc=$stamp"
    "$PGBIN/psql" "$DSN" -At -c "SELECT 'postgres='||version()"
    "$PGBIN/psql" "$DSN" -At -c "SELECT 'timescaledb='||extversion FROM pg_extension WHERE extname='timescaledb'"
    "$PGBIN/psql" "$DSN" -At -c "SELECT 'db_bytes_on_disk='||pg_database_size(current_database())"
    echo "dump_bytes=$(stat -f %z "$out")"
    echo "sha256=$(shasum -a 256 "$out" | cut -d' ' -f1)"
  } > "$STAGE/league_$stamp.manifest"
  echo "$stamp" > "$STAGE/LATEST"
  log "dump done: $(stat -f %z "$out") bytes; manifest $STAGE/league_$stamp.manifest"
}

sync_to() {
  dest="$1"
  [ -d "$dest" ] || die "destination '$dest' is not a directory (is the share mounted? see ls /Volumes)"
  [ -f "$STAGE/LATEST" ] || die "nothing staged; run 'dump' first"
  stamp=$(cat "$STAGE/LATEST")
  [ -f "$STAGE/league_$stamp.pgdump" ] || die "staged dump $stamp was already synced and removed; run 'dump' or 'all' for a fresh one"
  root="$dest/$SUB"
  mkdir -p "$root/postgres" "$root/data" "$root/code" 2>/dev/null \
    && touch "$root/.write_test" 2>/dev/null && rm -f "$root/.write_test" \
    || die "cannot write to $root (share mounted read-only? give your account Change permission on the Windows share, then remount)"
  log "destination $root"

  # 1. Postgres dump + globals + manifest
  for f in "league_$stamp.pgdump" "globals_$stamp.sql" "league_$stamp.manifest"; do
    if [ -f "$root/postgres/$f" ] && [ "$(stat -f %z "$root/postgres/$f")" = "$(stat -f %z "$STAGE/$f")" ]; then
      log "skip $f (already on the share with the same size)"; continue
    fi
    log "copy $f"
    cp "$STAGE/$f" "$root/postgres/$f.partial"
    mv "$root/postgres/$f.partial" "$root/postgres/$f"
  done
  want=$(sed -n 's/^dump_bytes=//p' "$STAGE/league_$stamp.manifest")
  got=$(stat -f %z "$root/postgres/league_$stamp.pgdump")
  [ "$want" = "$got" ] || die "size mismatch after copy: local $want vs share $got"
  "$PGBIN/pg_restore" --list "$root/postgres/league_$stamp.pgdump" > /dev/null || die "copied archive TOC unreadable"
  if [ "${VERIFY:-}" = "full" ]; then
    log "full verify: re-reading the dump from the share"
    wsha=$(sed -n 's/^sha256=//p' "$STAGE/league_$stamp.manifest")
    gsha=$(shasum -a 256 "$root/postgres/league_$stamp.pgdump" | cut -d' ' -f1)
    [ "$wsha" = "$gsha" ] || die "sha256 mismatch after copy"
  fi
  log "postgres dump verified ($got bytes)"

  # 2. data/ (models, scrapes, legacy data/league.db, logs). Incremental; never deletes on the share.
  #    Folders with thousands of small files crawl over SMB (several round trips per file), so any
  #    second-level folder with more than $TAR_MIN_FILES files is written as ONE tar under data/_tar/ instead.
  mkdir -p "$root/data/_tar"
  excl=""
  for d in "$REPO"/data/*/*/; do
    [ -d "$d" ] || continue
    rel=${d#"$REPO"/data/}; rel=${rel%/}
    n=$(find "$d" -type f | wc -l | tr -d ' ')
    if [ "$n" -gt "${TAR_MIN_FILES:-500}" ]; then
      name=$(printf '%s' "$rel" | tr '/' '_')
      log "tar $rel ($n files) -> data/_tar/$name.tar"
      tar -cf "$root/data/_tar/$name.tar.partial" -C "$REPO/data" "$rel"
      mv "$root/data/_tar/$name.tar.partial" "$root/data/_tar/$name.tar"
      excl="$excl --exclude=/$rel"
    fi
  done
  log "rsync data/ -> $root/data/ (excluding tarred folders)"
  # shellcheck disable=SC2086
  rsync -rlt --exclude='*.db-wal' --exclude='*.db-shm' --exclude='__pycache__' --exclude='.DS_Store' --exclude='*.tmp' $excl \
        "$REPO/data/" "$root/data/"

  # 3. code: every commit (incl. unpushed) as a git bundle + the working tree without data/ and .git/
  log "git bundle + working tree -> $root/code/"
  git bundle create "$root/code/esports2_$stamp.bundle" --all 2>/dev/null
  rsync -rlt --exclude='/data/' --exclude='/.git/' --exclude='/.venv/' --exclude='__pycache__' --exclude='.DS_Store' --exclude='exports/' \
        "$REPO/" "$root/code/worktree/"

  # 4. restore notes
  cat > "$root/README.txt" <<'TXT'
esports2 backup (made by scripts/backup_to_windows.sh on the MacBook Pro)

postgres/   league_<stamp>.pgdump  = pg_dump -Fc (zstd) of the live TimescaleDB database `league`
            globals_<stamp>.sql    = roles (pg_dumpall --globals-only)
            league_<stamp>.manifest= versions, sizes, sha256
data/       mirror of the repo's data/ folder (model artifacts, scrapes, legacy data/league.db SQLite, logs)
data/_tar/  folders with thousands of small files, each stored as one tar; restore with  tar -xf data/_tar/<name>.tar -C data
code/       esports2_<stamp>.bundle = all git commits (git clone esports2_<stamp>.bundle), worktree/ = working copy incl. uncommitted edits

Restore the database on a machine with PostgreSQL 18 and the SAME TimescaleDB version as in the manifest
(install that version first, ALTER EXTENSION timescaledb UPDATE afterwards if you want a newer one):
  createdb league
  psql league -c "CREATE EXTENSION timescaledb"
  psql league -c "SELECT timescaledb_pre_restore()"
  pg_restore -Fc -j 4 --no-owner -d league postgres/league_<stamp>.pgdump
  psql league -c "SELECT timescaledb_post_restore()"
Then point the Mac (or anything) at it with LOL_TICKER_DSN=postgresql://<host>:5432/league
TXT
  log "sync done -> $root"
  [ "${KEEP_STAGE:-}" = "1" ] || { rm -f "$STAGE/league_$stamp.pgdump"; log "removed local staged dump (KEEP_STAGE=1 keeps it)"; }
}

case "${1:-}" in
  dump) dump ;;
  sync) [ -n "${2:-}" ] || die "usage: sync <mounted-share-dir>"; sync_to "$2" ;;
  all)  [ -n "${2:-}" ] || die "usage: all <mounted-share-dir>"; dump; sync_to "$2" ;;
  *)    echo "usage: $0 dump | sync <dest> | all <dest>"; exit 2 ;;
esac
