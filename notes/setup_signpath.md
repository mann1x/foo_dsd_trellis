# SignPath Foundation setup for foo_dsd_trellis

The release workflow is wired to sign `foo_dsd_trellis.dll` and
`foo_dsd_trellis_worker.exe` via the SignPath Foundation OSS programme.
The signing steps activate automatically as soon as the GitHub repo
secrets below are set — until then the workflow keeps producing unsigned
binaries exactly like before.

This is a one-time setup. After it's done, every `git push origin v*`
tag will produce signed binaries automatically.

## 1. Apply for the SignPath Foundation OSS programme

Go to **https://signpath.org/apply** and submit the application:

- **Project name**: `foo_dsd_trellis`
- **Repository URL**: `https://github.com/mann1x/foo_dsd_trellis`
- **License**: (whichever the repo uses)
- **Brief description**: "DSD Trellis SDM DSP plugin for foobar2000.
  Provides high-quality DSD re-encoding, rate conversion, and GPU-
  accelerated room correction (convolution) at full DSD rates up to
  DSD512."

You'll be asked to sign in with GitHub so SignPath can verify you own
the repo. Approval typically takes **1-5 business days** — they manually
review that the project is genuinely open source and that builds are
reproducible from public source.

## 2. Once approved — create the project

After approval you'll have a SignPath organization. In the SignPath UI:

1. **Projects → Create new project**
   - Name: `foo_dsd_trellis`
   - Slug: `foo_dsd_trellis` (this slug is referenced in the workflow —
     keep it exactly as written or update the workflow to match)

2. **Project → Artifact Configurations → Create new**
   - Slug: `binaries` (also referenced in the workflow)
   - Configuration XML — paste this exactly:

   ```xml
   <?xml version="1.0" encoding="utf-8" ?>
   <artifact-configuration xmlns="http://signpath.io/artifact-configuration/v1">
     <zip-file>
       <pe-file path="foo_dsd_trellis.dll" />
       <pe-file path="foo_dsd_trellis_worker.exe" />
     </zip-file>
   </artifact-configuration>
   ```

   This tells SignPath that the input artifact is a zip containing two
   PE (Portable Executable) files at the root, and to sign both.

3. **Project → Trusted Build Systems → Create new**
   - Type: **GitHub Actions**
   - Repository owner: `mann1x`
   - Repository name: `foo_dsd_trellis`
   - Workflow file path: `.github/workflows/release.yml`
   - Branch / ref: leave the default (it'll use the tag that triggered
     the workflow)

4. **Project → Signing Policies → Create new**
   - Name: `Release signing`
   - Slug: `release-signing` (also referenced in the workflow)
   - Allowed trusted build systems: select the GitHub Actions one above
   - Approval: **Automatic** (for OSS) — SignPath verifies the build
     came from the trusted GitHub Actions workflow and signs without
     manual approval. Or you can require manual approval if you prefer
     to review every release.

## 3. Generate an API token

In the SignPath UI:

1. **User settings → API tokens → Create new token**
2. Scope: signing requests for the `foo_dsd_trellis` project
3. Save the token immediately — **it's shown only once**

## 4. Find your organization ID

Look at the URL when you're in your SignPath organization — it's the
GUID right after `/Web/`:

```
https://app.signpath.io/Web/<organization-id>/...
```

Copy that GUID.

## 5. Add the GitHub repo secrets

In `https://github.com/mann1x/foo_dsd_trellis/settings/secrets/actions`,
add two **Repository secrets** (not environment, not variables):

| Secret name | Value |
|---|---|
| `SIGNPATH_API_TOKEN` | the API token from step 3 |
| `SIGNPATH_ORGANIZATION_ID` | the GUID from step 4 |

That's it. The next `git push origin v*` tag will:

1. Build the binaries as usual
2. Stage `foo_dsd_trellis.dll` + `foo_dsd_trellis_worker.exe` into a
   temporary `signpath_input/` folder
3. Upload them as a GitHub Actions artifact
4. Submit a signing request to SignPath, passing the artifact ID
5. SignPath downloads the artifact, verifies it came from the trusted
   GitHub Actions workflow, signs both files, and uploads them back as
   a signed artifact
6. The workflow downloads the signed artifact, replaces the unsigned
   binaries with signed ones, and continues with packaging
7. The final `.fb2k-component` and `.zip` artifacts contain signed
   binaries

The workflow runs `Get-AuthenticodeSignature` on both files after the
replace step, so you'll see the signature details in the workflow logs.

## 6. Verify after first signed release

After the first signed release, do a sanity check:

```powershell
# Download and unzip the release
Expand-Archive foo_dsd_trellis-1.1.1-x64.zip
# Verify the binaries inside are signed
Get-AuthenticodeSignature foo_dsd_trellis-1.1.1-x64\foo_dsd_trellis.dll
Get-AuthenticodeSignature foo_dsd_trellis-1.1.1-x64\foo_dsd_trellis_worker.exe
```

Both should report `Status: Valid` and `SignerCertificate: SignPath
Foundation` (or whichever certificate SignPath uses for the OSS
programme).

You should also test on a Windows 11 machine with Smart App Control
enabled — the signed binaries should install without the SAC block.

## Troubleshooting

**"signing request rejected"** — check that the trusted build system
matches: branch, repo, workflow file path. SignPath verifies the
incoming GitHub Actions request came from exactly that source.

**"artifact configuration mismatch"** — the zip uploaded from the
workflow doesn't match the artifact configuration XML. Check the
filenames and paths. The configuration above expects `.dll` and `.exe`
at the zip root, no subdirectories.

**"timeout waiting for signing"** — the workflow waits up to 30 minutes
for SignPath to sign. If signing is set to manual approval, you may
need to log into SignPath and approve the request before the timeout
expires. Switch to automatic approval (step 2.4) to avoid this.

**Workflow runs without signing even though secrets are set** — the
`if: env.SIGNPATH_HAS_TOKEN == 'true'` condition checks via an env var
because GitHub doesn't allow `secrets.X != ''` in `if:` directly. Make
sure the secret name is exactly `SIGNPATH_API_TOKEN` with no typo.

## Cost

Free for OSS projects. SignPath Foundation is funded by the commercial
SignPath.io platform and provides certificates at no cost to verified
open-source projects. The only requirement is keeping the project
genuinely open source and not using the certificate to sign proprietary
or non-OSS code.

## References

- SignPath Foundation: https://signpath.org/
- OSS application: https://signpath.org/apply
- GitHub Action docs: https://github.com/SignPath/github-action-submit-signing-request
- Artifact configuration syntax: https://about.signpath.io/documentation/artifact-configuration
