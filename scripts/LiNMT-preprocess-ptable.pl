#!/usr/bin/perl

# * NiuTrans.NMT v1.0                   * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 10/30/2016                  * #
# * Time  : 15:31                       * #
# * NiuTrans.NMT-preprocess-ptable.pl   * #

use strict;


my $logo = "Preprocess lexical table\n";
print STDERR $logo;

my %param;

GetParameter(@ARGV);
PreprocessPhraseTable();



sub PreprocessPhraseTable 
{
  open (INPUT, "<", $param{"-input"}) or die "Error: can not read $param{'-input'}.\n";
  open (OUTPUT, ">", $param{"-output"}.".tmp1") or die "Error: can not write $param{'-output'}.tmp1.\n";
  print STDERR "\nSTEP 1. clean phrase table ...\n";
  my $line_num = 0;
  while (my $line = <INPUT>)
  {
    ++$line_num;

    if ($line_num % 10000 == 0)
    {
      print STDERR "\r  $line_num";
    }

    $line =~ s/[\r\n]//g;
    $line =~ s/\s+$//g;
    $line =~ s/^\s+//g;
    
    
    
    my @fields = split / \|\|\| /, $line;
    if (scalar(@fields) < 3)
    {
      print STDERR "\n  Format error in $line_num\n";
      next;
    }
    
    my @src_words = split /\s+/, $fields[0];
    if ($fields[1] eq "<NULL>")
    {
      next;
    }
    if (scalar(@src_words) > 1)
    {
      next;
    }
    else
    {
      #print OUTPUT $line."\n";
      my @scores = split /\s+/, $fields[2];
      print OUTPUT $fields[0]." ||| ".$fields[1]." ||| ".$scores[0]."\n";
    }
  }
  print STDERR "\r  $line_num\n";

  close (INPUT);
  close (OUTPUT);
  
=pod
  print STDERR "\nSTEP 1. Invert src & tgt fileds ...\n";
  my $line_num = 0;
  while (my $line = <INPUT>) {
    ++$line_num;
    $line =~ s/[\r\n]//g;
    $line =~ s/\s+$//g;
    $line =~ s/^\s+//g;
    
    my @fields = split /\s/, $line;
    if (scalar(@fields) ne 3) {
      print STDERR "\n  Format error in $line_num\n";
      next;
    }
    print OUTPUT $fields[1]." ".$fields[0]." ".$fields[2]."\n";
    if ($line_num % 10000 == 0)
    {
      print STDERR "\r  $line_num";
    }
  }
  print STDERR "\r  $line_num\n";
=cut

  
  print STDERR "\nSTEP 2. sort file ...\n";
  Ssystem("LC_ALL=C sort $param{'-output'}.tmp1 > $param{'-output'}.tmp1.sort");
  
  
  print STDERR "STEP 3. Reserve only top 1 ...\n";
  open (INPUT, "<", $param{"-output"}.".tmp1.sort") or die "Error: can not read $param{'-output'}.tmp1.sort.\n";
  open (OUTPUT, ">", $param{"-output"}) or die "Error: can not write $param{'-output'}.\n";
  
  my $line_num = 0;
  my $last_src = "";
  my $first_line_mode = 1;
  my $cur_src = "";
  my $cur_tgt = "";
  my $cur_sco = 0;
  while (my $line = <INPUT>)
  {
    ++$line_num;
    if ($line_num % 10000 == 0)
    {
      print STDERR "\r  $line_num";
    }
  
  
    $line =~ s/[\r\n]//g;
    $line =~ s/\s+$//g;
    $line =~ s/^\s+//g;
    
    my @fields = split / \|\|\| /, $line;
    if (scalar(@fields) ne 3)
    {
      print STDERR "\n  Format error in $line_num\n";
    }
    
    if ($first_line_mode eq 1) 
    {
      $cur_src = $fields[0];
      $cur_tgt = $fields[1];
      $cur_sco = $fields[2];
      $last_src = $cur_src;
      $first_line_mode = 0;
      next;
    } 
    else 
    {
    
      if (($fields[0] eq $last_src)) 
      {
        if (($fields[2] > $cur_sco))
        {
          $cur_tgt = $fields[1];
          $cur_sco = $fields[2];
        }
        else
        {
          next;
        }
      }
      else
      {
        print OUTPUT $cur_src." ||| ".$cur_tgt." ||| ".$cur_sco."\n";
        $last_src = $fields[0];
        $cur_src = $fields[0];
        $cur_tgt = $fields[1];
        $cur_sco = $fields[2];
      }
    }
  }
  print OUTPUT $cur_src." ||| ".$cur_tgt." ||| ".$cur_sco."\n";
  print STDERR "\r  $line_num\n";
  
  
  close (INPUT);
  close (OUTPUT);
  
  unlink $param{"-output"}.".tmp1";
  unlink $param{"-output"}.".tmp1.sort";
}



sub GetParameter
{
  if( ( scalar( @_ ) < 2 ) || ( scalar( @_ ) % 2 != 0 ) )
  {
    print STDERR "[USAGE]\n".
                 "        NiuTrans.NMT-preprocess-ptable.pl   [OPTIONS]\n".
                 "[OPTION]\n".
                 "            -input : Input phrase table\n".
                 "           -output : Preprocessed phrase table\n".
                 "[EXAMPLE]\n".
                 "        perl NiuTrans.NMT-preprocess-ptable.pl\n";
    exit( 0 );
  }
          
  my $pos;
  for( $pos = 0; $pos < scalar( @_ ); ++$pos )
  {
    my $key = $ARGV[ $pos ];
    ++$pos;
    my $value = $ARGV[ $pos ];
    $param{ $key } = $value;
  }

  if (!exists $param{"-input"} || !exists $param{"-output"})
  {
    print STDERR "ERROR: '-config', '-input', or '-output' must be assigned!\n";
    exit(1);
  }
}



sub Ssystem
{
  print STDERR "Running: @_\n";
  system( @_ );
  if( $? == -1 )
  {
    print STDERR "Error: Failed to execute: @_\n  $!\n";
    exit( 1 );
  }
  elsif( $? & 127 )
  {
    printf STDERR "Error: Execution of: @_\n   die with signal %d, %s coredump\n",
                  ($? & 127 ), ( $? & 128 ) ? 'with' : 'without';
    exit( 1 );
  }
  else
  {
    my $exitcode = $? >> 8;
    print STDERR "Exit code: $exitcode\n" if $exitcode;
    return ! $exitcode;
  }         
}



