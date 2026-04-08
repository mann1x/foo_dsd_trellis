# Microsoft Trusted Signing setup for foo_dsd_trellis

The release workflow is wired to sign `foo_dsd_trellis.dll` and
`foo_dsd_trellis_worker.exe` via Microsoft Trusted Signing (formerly
Azure Code Signing). The signing steps activate automatically as soon
as the GitHub repo secrets below are set — until then the workflow
keeps producing unsigned binaries exactly like before.

This is a one-time setup, then every `git push origin v*` tag produces
signed binaries automatically. The same Trusted Signing account can be
reused for every other Windows project you build (each project just
needs the same secrets configured in its repo, or shared via a GitHub
organization secret).

## Cost

- **Basic plan**: **$9.99 USD/month** (~$120/year). Includes **5,000 signing
  operations per month**, more than enough for any indie OSS project.
- Premium plan: ~$100/month, only needed for high-volume CI signing
  (>5,000 ops/month). Skip it.

Identity validation is **free** under the Basic plan — Microsoft verifies
your government-issued ID once and you're set.

## 1. Azure prerequisites

You need:

- **An Azure subscription**. Free tier is fine — Trusted Signing isn't
  covered by free credits, but you only need a valid payment method on
  file. If you don't have one yet, sign up at https://azure.microsoft.com/free.
- **A working Azure CLI** (optional but makes the setup faster):
  https://learn.microsoft.com/cli/azure/install-azure-cli-windows

## 2. Register the Microsoft.CodeSigning resource provider

In the Azure Portal: **Subscriptions → (your sub) → Resource providers →
search "Microsoft.CodeSigning" → Register**.

Or via Azure CLI:

```powershell
az provider register --namespace Microsoft.CodeSigning
```

Wait until the state shows "Registered" (1-2 minutes).

## 3. Create a Trusted Signing account

In the Azure Portal: **Create a resource → search "Trusted Signing
Account" → Create**.

- **Subscription**: your subscription
- **Resource group**: create a new one, e.g. `rg-codesigning`
- **Account name**: e.g. `mann1x-signing` (lowercase, 3-24 chars,
  alphanumeric + hyphens)
- **Region**: pick the **closest** region — this affects the endpoint
  URL and signing latency. For Italy / Western Europe, pick
  **West Europe** (`weu`). Other options:
  - East US: `eus`
  - West Central US: `wcus`
  - West Europe: `weu`
  - North Europe: `neu`
- **Pricing tier**: **Basic** ($9.99/month)
- Click **Review + Create** → **Create**

Note your endpoint URL — it's `https://<region-code>.codesigning.azure.net/`.
For West Europe: `https://weu.codesigning.azure.net/`.

## 4. Identity validation

Inside the Trusted Signing account in the portal: **Settings → Identity
validation → New identity validation request**.

Pick **Individual** validation (free under Basic plan, suitable for
indie devs). The cert subject will be `Open Source Developer, Your Name`.

You'll be asked to upload:
- Government-issued photo ID (passport / national ID card / driver's license)
- Sometimes a second proof (utility bill / bank statement) — depends on
  region

Microsoft's review takes **1-3 business days**. You can't proceed to
step 5 until the request is **Completed**.

> **If you want to sign as a company instead** (e.g., your business
> entity), pick **Public** validation. Requires a registered legal entity
> (LLC / Ltd / SRL). The cert subject will be the company name. Costs
> the same as Individual under the Basic plan, but the verification is
> stricter.

## 5. Create a Certificate Profile

After identity validation completes: **Settings → Certificate profiles
→ New certificate profile**.

- **Profile name**: e.g. `release-signing`
- **Profile type**: **Public Trust** (issues a publicly-trusted cert)
- **Identity validation**: select the one you just created
- **Include city / state / country in subject**: optional, your choice

Click **Create**. The cert is auto-issued by the Trusted Signing
service — no additional fees, no waiting. The profile name is the
value you'll set as `AZURE_SIGNING_PROFILE` later.

## 6. Create an Entra ID App Registration (service principal)

The GitHub Actions workflow needs Azure credentials to call the
Trusted Signing API. We use a Service Principal with client-secret auth.

In the Azure Portal: **Microsoft Entra ID → App registrations →
New registration**.

- **Name**: e.g. `github-actions-signing`
- **Supported account types**: "Accounts in this organizational directory only"
- **Redirect URI**: leave blank
- Click **Register**

After registration, on the App's **Overview** page, copy:
- **Application (client) ID** → this is `AZURE_CLIENT_ID`
- **Directory (tenant) ID** → this is `AZURE_TENANT_ID`

Then **Certificates & secrets → Client secrets → New client secret**:
- **Description**: `github-actions`
- **Expires**: 24 months (max — you'll need to rotate before expiry)
- Click **Add**, then **immediately copy the Value column** (it's only
  shown once). This is `AZURE_CLIENT_SECRET`.

## 7. Grant the App the signing role

The Service Principal needs permission to use the certificate profile
for signing. The role is **Trusted Signing Certificate Profile Signer**.

In the Trusted Signing account: **Access control (IAM) → Add →
Add role assignment**.

- **Role**: search for and select **Trusted Signing Certificate Profile Signer**
- **Assign access to**: User, group, or service principal
- **Members**: click **Select members**, search for your App
  Registration name (`github-actions-signing`), select it
- Click **Review + assign**

Wait ~1-2 minutes for the role assignment to propagate.

## 8. Add the GitHub repo secrets

In `https://github.com/mann1x/foo_dsd_trellis/settings/secrets/actions`,
add **Repository secrets** (not environment, not variables):

| Secret name | Value | Where it comes from |
|---|---|---|
| `AZURE_TENANT_ID` | the Directory (tenant) ID | App Registration → Overview |
| `AZURE_CLIENT_ID` | the Application (client) ID | App Registration → Overview |
| `AZURE_CLIENT_SECRET` | the client secret value | App Registration → Certificates & secrets (only shown once at creation) |
| `AZURE_SIGNING_ENDPOINT` | e.g. `https://weu.codesigning.azure.net/` | The endpoint URL for your Trusted Signing region |
| `AZURE_SIGNING_ACCOUNT` | e.g. `mann1x-signing` | The Trusted Signing account name from step 3 |
| `AZURE_SIGNING_PROFILE` | e.g. `release-signing` | The certificate profile name from step 5 |

> **Reuse for other projects**: every other Windows repo of yours that
> needs signing just needs the **same six secrets** added — same
> Trusted Signing account, same cert profile, same SP. No per-project
> Azure setup. If you have a GitHub organization, you can set them as
> **Organization secrets** once and they're available to every repo.

## 9. Test it

Push any tag matching `v*` (e.g. re-tag the current release):

```bash
git tag -d v1.1.0
git push --delete origin v1.1.0
git tag v1.1.0
git push origin v1.1.0
```

The workflow will:

1. Build as usual
2. Hit the `Sign binaries (Microsoft Trusted Signing)` step (no longer
   skipped because `AZURE_CLIENT_ID` is set)
3. Call the Trusted Signing API with your credentials, which signs
   `foo_dsd_trellis.dll` and `foo_dsd_trellis_worker.exe` in place
4. Run the `Verify signatures` step which calls
   `Get-AuthenticodeSignature` on both files and **fails the build if
   either signature is invalid**
5. Continue with packaging — the `.fb2k-component` and `.zip` artifacts
   now contain signed binaries
6. Publish the GitHub release with the signed artifacts

## 10. Verify after first signed release

```powershell
# Download and unzip the release
Expand-Archive foo_dsd_trellis-1.1.0-x64.zip
# Verify signatures
Get-AuthenticodeSignature foo_dsd_trellis-1.1.0-x64\foo_dsd_trellis.dll
Get-AuthenticodeSignature foo_dsd_trellis-1.1.0-x64\foo_dsd_trellis_worker.exe
```

Both should report:
- `Status: Valid`
- `SignerCertificate: ...` with subject containing `Open Source Developer, <your name>`
- `TimeStamperCertificate: ...` (timestamp from `timestamp.acs.microsoft.com`)

Then test on a Windows 11 machine with **Smart App Control enabled** —
the signed binaries should install without the SAC block.

> **SmartScreen reputation note**: even with a valid signature, fresh
> certificates take a few hundred installs before SmartScreen learns
> they're trusted. During that period users may still see "Windows
> protected your PC" with a "Run anyway" button. Once you reach the
> reputation threshold, the warning disappears. **SAC accepts the
> signature immediately** because it's chained to a Microsoft root.

## Troubleshooting

**`401 Unauthorized` from Trusted Signing API**
- Check that the role assignment in step 7 actually went through
  (sometimes takes 5+ minutes to propagate)
- Verify the App Registration has the role on the **Trusted Signing
  account**, not on a wrong scope
- Confirm `AZURE_TENANT_ID` and `AZURE_CLIENT_ID` aren't swapped (easy
  mistake — both are GUIDs)

**`certificate-profile-name not found`**
- The profile name is case-sensitive
- It must be created **inside the Trusted Signing account** referenced
  in `AZURE_SIGNING_ACCOUNT`, not in a different account
- Identity validation must be **Completed** before the profile can sign

**`endpoint not reachable` / DNS errors**
- The endpoint is region-specific. `https://weu.codesigning.azure.net/`
  for West Europe, etc. Check the region of your Trusted Signing account
  and use the matching endpoint.

**Signature verifies but SmartScreen still shows the warning**
- Expected for a fresh certificate. Reputation builds over the first
  few hundred installs. Nothing to fix on your side.

**Signature verifies but Smart App Control still blocks**
- Should not happen — SAC trusts the Microsoft root immediately
- Double-check by running `Get-AuthenticodeSignature` on the **deployed**
  binary (the one inside the `.fb2k-component`), not just the build
  output. It's possible the signing step ran before the file was copied
  into the staging directory.

**Client secret expired**
- Azure App Registration client secrets max out at 24 months
- Calendar a reminder for ~22 months from creation
- When it expires, generate a new one in step 6 and update the
  `AZURE_CLIENT_SECRET` repo secret

## Reusing for other projects

For each additional Windows project you want signed:

1. Add the same six secrets to the new repo (or use a GitHub
   organization secret to share once)
2. Add the same `Sign binaries (Microsoft Trusted Signing)` and
   `Verify signatures` steps to that repo's release workflow,
   pointing `files-folder` at wherever that project's binaries live
3. That's it — no Azure setup, no new App Registration, no new role
   assignments. The Trusted Signing account, certificate profile, and
   service principal are all reusable across projects.

The $9.99/month pays for **everything you sign**, not per-project.

## References

- Microsoft Trusted Signing docs: https://learn.microsoft.com/azure/trusted-signing/
- Quickstart: https://learn.microsoft.com/azure/trusted-signing/quickstart
- GitHub Action: https://github.com/Azure/trusted-signing-action
- Pricing: https://azure.microsoft.com/pricing/details/trusted-signing/
- Identity validation requirements: https://learn.microsoft.com/azure/trusted-signing/concept-trusted-signing-identity-verification
